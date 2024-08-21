
import os
from typing import Annotated, TypedDict, Any, Dict, Optional, Sequence, Type, Union
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool
from langsmith import Client
import json
from sqlalchemy.engine import Result

# from utils import generate_mermaid


##########################
### Set up llm
##########################
_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
experiment_prefix="sql-agent-gpt4o"
metadata = "Snowflake, gpt-4o base-case-agent"

##########################
### Set up db
##########################

username = os.environ['DATABASE_USERNAME']
password = os.environ['DATABASE_PASSWORD']
snowflake_account = os.environ['DATABASE_ACCOUNT']
warehouse = os.environ['DATABASE_WAREHOUSE']
role = os.environ['DATABASE_ROLE']
database = os.environ['DATABASE_NAME']
schema = os.environ['DATABASE_SCHEMA']

snowflake_url = f"snowflake://{username}:{password}@{snowflake_account}/{database}/{schema}?warehouse={warehouse}&role={role}"
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri(snowflake_url, 
                          sample_rows_in_table_info=1, 
                          include_tables=['vw_retail_transactions'], 
                          view_support=True)

# # we can see what information is passed to the LLM regarding the database
# print(db.dialect)
# print(db.get_table_info())
# print(db.get_usable_table_names())
# # db.run("SELECT * FROM vw_retail_transactions LIMIT 10;")

##########################
### tools
##########################

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

class _FormatOutputToolInput(BaseModel):
    text: str = Field(description="Interpretation of the user question")
    sql: str = Field(description="Syntactically correct snowflake query which can be used to answer the user's question")


class FormatOutputTool(BaseTool):
    """Tool to parse output in expected format."""

    name: str = "output_formatter"
    description: str = """
    Make sure this tool is used after the tool sql_db_query_checker
    """
    args_schema: Type[BaseModel] = _FormatOutputToolInput

    def _run(
        self,
        text: str,
        sql: str
    ) -> Any:
        """Output things in expected format."""
        return json.dumps({'text': text, 'sql': sql})

format_output_tool = FormatOutputTool()

# Exclude the QuerySQLDataBaseTool as we don't want to run the query here
tools = tools[1: ]
tools.append(format_output_tool)
llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


##########################
### state machine
##########################
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    generated_flag: bool = False
    generated_sql: str = None
    generated_explanation: str = None


##########################
### workflow
##########################
class TextToSqlAgent():

    def __init__(self):

        workflow = StateGraph(State)
        tool_node = ToolNode(tools)

        # Define the node
        def call_model(state: State):

            SQL_PREFIX = """
            You are an agent designed to interact with a SQL database.

            Given an input question, create a syntactically correct {dialect} query to run, then return suggested query along with your interpretation of this question. 

            DO NOT try executing the query, just validate the suggested query statement as is, along with your interpretation.

            Notes:
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results using the LIMIT clause.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
            If the question does not seem related to the database, just return "Your input seems not related to the database. Execuse me for not answering." as the answer.
            When selecting from a table, ALWAYS use format "{database_name}.{schema_name}.table_name" rather than the short "table_name".

            You have access to tools for interacting with the database.
            Only use the below tools.
            Only use the information returned by the below tools to construct your final answer.

            Before generating the query, ALWAYS call tool sql_db_list_tables to list out tables available.
            Before generating the query, ALWAYS call tool sql_db_schema to understand the metadata, schema and sample values.
            After generating the query, ALWAYS validate it using the tool sql_db_query_checker.
            After calling tool sql_db_query_checker, ALWAYS call the tool output_formatter.
            """

            messages = state["messages"]

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SQL_PREFIX),
                    ("placeholder", "{chat_history}")
                ]
            )
            chain = (
                prompt
                | llm_with_tools
            )
            response = chain.invoke({"chat_history": messages, "dialect" : "snowflake", "top_k": 5, "database_name": database, "schema_name": schema})

            # print("=====response=====")
            # print(response)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call["name"] == "output_formatter":
                    return { 
                        "messages": [response], 
                        "generated_flag": True, 
                        "generated_sql": tool_call['args']['sql'], 
                        "generated_explanation": tool_call['args']['text'] 
                    }

            return { "messages": [response] }


        def should_continue_or_end(state: State):
            messages = state["messages"]
            last_ai_message = messages[-1]

            if last_ai_message.tool_calls:
                return "tools"
            
            return END

        # Add the node to the flow
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # Add edges to the flow
        workflow.add_edge(START, 'agent')
        workflow.add_conditional_edges('agent', should_continue_or_end, {"tools": "tools", END: END})
        workflow.add_edge('tools', 'agent')

        # Initialize memory to persist state between graph runs
        checkpointer = MemorySaver()

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable.
        # Note that we're (optionally) passing the memory when compiling the graph
        self.app = workflow.compile(checkpointer=checkpointer)

        # generate_mermaid(self.app, 'images/output_image_text_to_sql.png')


    def predict(self, query: str, thread_id: int) -> str:
        final_state = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        ai_msg = final_state['messages'][-1].content
        
        analyst_msg = {
                    "role": "analyst",
                    "content": []
                }
        
        if 'generated_flag' in final_state and final_state['generated_flag']:
            sql = final_state['generated_sql']
            text = final_state['generated_explanation']
            analyst_msg["content"].append({"type": "text", "text": text})
            analyst_msg["content"].append({"type": "sql", "statement": sql})
        else:
            analyst_msg["content"].append({"type": "text", "text": ai_msg})

        results = {}
        results["message"] = analyst_msg

        # print("=====final_state=====")
        # for msg in final_state["messages"]:
        #     # msg.pretty_print()
        #     print(msg)
        # print(final_state["sql_suggested_flag"])
        # print(final_state["sql_suggested"])
        # print("\n")

        return results
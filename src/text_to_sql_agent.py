
from typing import Annotated, TypedDict
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langsmith import Client
from PIL import Image as PILImage
import io
import json


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

with open("creds.json") as f:
    config = json.load(f)

username = config["username"]
password = config["password"]
snowflake_account = config["account"]
warehouse = config["warehouse"]
role = config["role"]
database = config["database"]
schema = config["schema"]

snowflake_url = f"snowflake://{username}:{password}@{snowflake_account}/{database}/{schema}?warehouse={warehouse}&role={role}"

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

llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


##########################
### state machine
##########################
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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
            Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results using the LIMIT clause.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools.
            Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            If the question does not seem related to the database, just return "I don't know" as the answer.
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
            response = chain.invoke({"chat_history": messages, "dialect" : "snowflake", "top_k": 5})

            # if (response.tool_calls[0].name === 'sql') {
            #     reutn 
            # }
            
            # # update state with ai message
            # messages.append(response)

            return { "messages": [response],  }


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


    def predict(self, query: str, thread_id: int) -> str:
        final_state = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )

        # for msg in final_state['messages']:
        #     print(msg)
        #     print('\n')
        #     print('\n')

        return final_state['messages'][-1].content

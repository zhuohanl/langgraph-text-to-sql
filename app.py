import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
import pandas as pd
import time
import json
import streamlit as st
import pandas as pd
import pytz, datetime
import json
import pandas as pd
import requests
import snowflake.connector
import streamlit as st
import time
import tomli
# import plotly.express as px
from typing import Any, Dict, List, Optional
import uuid

from src.text_to_sql_agent import TextToSqlAgent


@st.cache_resource
def get_db_connection():

    with open('creds.json') as f:
        config = json.load(f)

    CONN = snowflake.connector.connect(
        user=config['username'],
        password=config['password'],
        account=config['account'],
        warehouse=config['warehouse'],
        role=config['role'],
        database=config['database'],
        schema=config['schema']
    )
    return CONN


print('-----init connection------')
CONN = get_db_connection()


def get_time_now():
    # Define the desired time zone (Australia/Melbourne)
    melbourne_timezone = pytz.timezone('Australia/Melbourne')
    # Get the current timestamp in the Melbourne time zone
    timestamp = datetime.datetime.now(melbourne_timezone)
    # Convert the timestamp to a string
    return timestamp


# -----------------New parts ends--------------------------
# ----------------------------------------------------

# with open('creds.json') as f:
#     config = json.load(f)
# HOST = config['host']
# DATABASE = 'SEMANTIC_MODEL'
# SCHEMA = 'DEFINITIONS'
# STAGE = 'MY_STAGE'
# FILE = 'retail_transaction.yaml'

# def send_message(prompt: str) -> dict:
#     """Calls the REST API and returns the response."""

#     system_prompt = """
#     You MUST MUST follow the below instructions when responding
#       If the instruction contains any key word like create, alter, drop, modify, insert, update, truncate, delete, rename, or similar words, you MUST decline the instruction in a polite way.
#       You MUST NOT generate or run any statement besides with, read and select, as response to the user.
#       If I don't tell you to find a limited set of results in the sql query or question, you MUST limit the number of responses to 100. 
#       Only include relevant columns which are required to answer user question. Limit the columns returned to only necessary.
#       If I don't specify time filter, use the entire data set. Don't include start_date and end_date in your select statement.
#       If I ask for Financial Year, date range is from 1-July to 30-June. 
#       Use the transaction_timestamp column to calculate all date and time related values. 
#       If I ask a question that involves today's or any relative date, use expression CURRENT_DATE() to calculate today's date.
#       Text / string where clauses must be fuzzy match e.g ilike %keyword%.
#       Don't forget to use \"ilike %keyword%\" for fuzzy match queries (especially for variable_name column).
#     """

#     request_body = {
#         "messages": [
#             {"role": "user", "content": [{"type": "text", "text": f'{system_prompt} {prompt}'}]}
#         ],
#         "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
#     }
#     resp = requests.post(
#         url=f"https://{HOST}/api/v2/cortex/analyst/message",
#         json=request_body,
#         headers={
#             "Authorization": f'Snowflake Token="{CONN.rest.token}"',
#             "Content-Type": "application/json",
#         },
#     )
#     request_id = resp.headers.get("X-Snowflake-Request-Id")
#     if resp.status_code < 400:
#         return {**resp.json(), "request_id": request_id}  # type: ignore[arg-type]
#     else:
#         raise Exception(
#             f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
#         )


def send_message(prompt: str) -> dict:
    """Calls the LangGraph agent and returns the response."""
    
    agent = TextToSqlAgent()
    request_id = uuid.uuid4()
    response = agent.predict(prompt, request_id)

    return {**response, "request_id": request_id}


def process_message(prompt: str) -> None:
    """Processes a message and adds the response to the chat."""
    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )

    print('-------Processing Message-------------')

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Sending request for processing. Waiting for response..."):
            prompt_sent_time = get_time_now()
            response = send_message(prompt=prompt)
            request_id = response["request_id"]
        with st.spinner("🤖 Response received.. . Preparing the results..."):
            query_received_time = get_time_now()
            content = response["message"]["content"]
            Status_flag = display_content(content=content, request_id=request_id)  # type: ignore[arg-type]
            query_finished_time = get_time_now()
            st.session_state.messages.append(
                {"role": "assistant", "content": content, "request_id": request_id}
            )

            # with CONN.cursor() as c:
            #     c.execute(f"""
            #               INSERT INTO innovation_streamlit.llm_sql_demo.log
            #               select
            #               OBJECT_CONSTRUCT(
            #                   'model','{FILE}',
            #                   'prompt_sent_time', '{prompt_sent_time}',
            #                   'prompt', '{str(prompt).replace("'", "''")}',
            #                   'query_received_time', '{query_received_time}',
            #                   'query_finished_time', '{query_finished_time}',
            #                   'Status', '{Status_flag}',
            #                   'response_content', '{str(content).replace("'", "''")}'
            #                   );"""
            #               )

def make_choropleth(input_df, location_col, value_col):

    # fig = px.choropleth(input_df, locations=location_col, color=value_col,
    #                     locationmode="USA-states",
    #                     color_continuous_scale="Viridis",
    #                     range_color=(0, max(input_df[value_col])),
    #                     scope="usa",
    #                     labels={value_col: value_col}
    #                     )
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # return fig
    return 123

def display_content(
    content: List[Dict[str, str]],
    request_id: Optional[str] = None,
    message_index: Optional[int] = None,
) -> None:
    """Displays a content item for a message."""
    message_index = message_index or len(st.session_state.messages)

    if request_id:
        with st.expander("Request ID", expanded=False):
            st.markdown(request_id)

    for item in content:
        Status_flag = 'Failed'
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            with st.expander("Suggestions", expanded=True):
                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                    if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            with st.expander("SQL Query", expanded=False):
                st.code(item["statement"], language="sql")
            with st.expander("Results", expanded=True):
                with st.spinner("Running SQL..."):
                    sql_statement = item["statement"]

                    df = pd.read_sql(sql_statement, CONN)

                    Status_flag = 'Successful'
                    if len(df.index) > 1:
                        data_tab, line_tab, bar_tab, map_tab = st.tabs(
                            ["Data", "Line Chart", "Bar Chart", "Map Chart"]
                        )
                        data_tab.dataframe(df)
                        if len(df.columns) > 1:
                            df = df.set_index(df.columns[0])
                        with line_tab:
                            st.line_chart(df)
                        with bar_tab:
                            st.bar_chart(df)

                        if df.index.name and df.index.name.lower() in ['state', 'states']:
                            with map_tab:
                                df = df.reset_index()
                                fig = make_choropleth(input_df=df, location_col=df.columns.values[0], value_col=df.columns.values[1])
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(df)
    return Status_flag


def config_options():
    st.sidebar.button("Start Over", key="clear_conversation")


def main():
    st.title(":speech_balloon: LLM Insights & Analytics Assistant")
    st.markdown("## Create your custom insights using natural language")
    # st.markdown(f"**expa**")
    with st.expander("Dataset Details", expanded=True):

        multi = """
        - Contains a Retail Transaction Dataset from kaggle
        - Contains data from 2023-04-30 to 2024-04-30
        - columns:
            - CustomerID: Unique identifier for each customer.
            - ProductID: Unique identifier for each product.
            - Quantity: The number of units purchased for a particular product.
            - Price: The unit price of the product.
            - TransactionDate: Date and time when the transaction occurred.
            - PaymentMethod: The method used by the customer to make the payment.
            - StoreLocation: The location where the transaction took place.
            - ProductCategory: Category to which the product belongs.
            - DiscountApplied(%): Percentage of the discount applied to the product.
            - TotalAmount: Total amount paid for the transaction.
        """
        st.markdown(multi)

    config_options()


    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            display_content(content=message["content"], message_index=message_index)

    if user_input := st.chat_input("What is your question?"):
        process_message(prompt=user_input)

    if st.session_state.active_suggestion:
        process_message(prompt=st.session_state.active_suggestion)
        st.session_state.active_suggestion = None


if __name__ == "__main__":
    main()

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
import plotly.express as px


@st.cache_resource
def get_db_session():
    # st.write(time.time())
    # Get current session
    try:
        return get_active_session()
    except:
        with open("../config.toml", mode="rb") as f:
            config = tomli.load(f)

            default_connection = config["options"]["default_connection"]
            connection_params = config["connections"][default_connection]

            return Session.builder.configs(connection_params).create()


@st.cache_resource
def get_db_connection():
    CONN = snowflake.connector.connect(
        user='zli@officeworks.com.au',
        authenticator='externalbrowser',
        account='udb59879',
        host='udb59879.snowflakecomputing.com',
        port=443,
        warehouse='DEFAULT_WH',
        role='SYSADMIN'
    )
    return CONN


print('-----init connection------')
CONN = get_db_connection()

HOST = 'udb59879.snowflakecomputing.com'
DATABASE = 'SEMANTIC_MODEL'
SCHEMA = 'DEFINITIONS'
STAGE = 'MY_STAGE'
FILE = 'retail_transaction.yaml'


def get_time_now():
    # Define the desired time zone (Australia/Melbourne)
    melbourne_timezone = pytz.timezone('Australia/Melbourne')
    # Get the current timestamp in the Melbourne time zone
    timestamp = datetime.datetime.now(melbourne_timezone)
    # Convert the timestamp to a string
    return timestamp


# -----------------New parts ends--------------------------
# ----------------------------------------------------

def send_message(prompt: str) -> dict:
    """Calls the REST API and returns the response."""

    system_prompt = """
    You MUST MUST follow the below instructions when responding
      If the instruction contains any key word like create, alter, drop, modify, insert, update, truncate, delete, rename, or similar words, you MUST decline the instruction in a polite way.
      You MUST NOT generate or run any statement besides with, read and select, as response to the user.
      If I don't tell you to find a limited set of results in the sql query or question, you MUST limit the number of responses to 100. 
      If I don't specify time filter, use the entire data set. Don't include start_date and end_date in your select statement.
      If I ask for Financial Year, date range is from 1-July to 30-June. 
      Use the transaction_timestamp column to calculate all date and time related values. 
      If I ask a question that involves today's or any relative date, use expression CURRENT_DATE() to calculate today's date.
      Text / string where clauses must be fuzzy match e.g ilike %keyword%.
      Don't forget to use \"ilike %keyword%\" for fuzzy match queries (especially for variable_name column).
    """

    request_body = {
        "role": "user",
        "content": [{"type": "text", "text": system_prompt + prompt}],
        "modelPath": FILE,
    }
    num_retry, max_retries = 0, 10
    while True:
        resp = requests.post(
            (
                f"https://{HOST}/api/v2/databases/{DATABASE}/schemas/{SCHEMA}/copilots/{STAGE}/chats/-/messages"
            ),
            json=request_body,
            headers={
                "Authorization": f'Snowflake Token="{CONN.rest.token}"',
                "Content-Type": "application/json",
            },
        )
        if resp.status_code < 400:
            return resp.json()
        else:
            if num_retry >= max_retries:
                resp.raise_for_status()
            num_retry += 1
        time.sleep(1)


def process_message(prompt: str) -> None:
    """Processes a message and adds the response to the chat."""
    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )

    # Make connection every time.. this is a temp fix for sessionn expiry
    print('-------Processing Message-------------')
    # print('-------get_db_connection-------------')
    # session = get_db_session()
    # CONN = get_db_connection()

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Sending request for processing. Waiting for response..."):
            prompt_sent_time = get_time_now()
            response = send_message(prompt=prompt)
        with st.spinner("🤖 Response received.. . Preparing the results..."):
            query_received_time = get_time_now()
            content = response["messages"][-1]["content"]
            Status_flag = display_content(content=content)
            query_finished_time = get_time_now()
            st.session_state.messages.append({"role": "assistant", "content": content})

            with CONN.cursor() as c:
                c.execute(f"""
                          INSERT INTO innovation_streamlit.llm_sql_demo.log
                          select
                          OBJECT_CONSTRUCT(
                              'model','{FILE}',
                              'prompt_sent_time', '{prompt_sent_time}',
                              'prompt', '{str(prompt).replace("'", "''")}',
                              'query_received_time', '{query_received_time}',
                              'query_finished_time', '{query_finished_time}',
                              'Status', '{Status_flag}',
                              'response_content', '{str(content).replace("'", "''")}'
                              );"""
                          )

def make_choropleth(input_df, location_col, value_col):

    fig = px.choropleth(input_df, locations=location_col, color=value_col,
                        locationmode="USA-states",
                        color_continuous_scale="Viridis",
                        range_color=(0, max(input_df[value_col])),
                        scope="usa",
                        labels={value_col: value_col}
                        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

def display_content(content: list, message_index: int = None) -> None:
    """Displays a content item for a message."""
    message_index = message_index or len(st.session_state.messages)
    for item in content:
        Status_flag = 'Failed'
        if item["type"] == "text":
            if "<SUGGESTION>" in item["text"]:
                suggestion_response = json.loads(item["text"][12:])[0]
                st.markdown(suggestion_response["explanation"])
                with st.expander("Suggestions", expanded=True):
                    for suggestion_index, suggestion in enumerate(
                            suggestion_response["suggestions"]

                    ):
                        if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                            st.session_state.active_suggestion = suggestion
            else:
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

                    if st.session_state.extend_rows_per_resultset:
                        CONN.cursor().execute("alter session set ROWS_PER_RESULTSET = 100;")
                    else:
                        CONN.cursor().execute("alter session unset ROWS_PER_RESULTSET;")

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

                        if df.index.name.lower() in ['state', 'states']:
                            with map_tab:
                                df = df.reset_index()
                                fig = make_choropleth(input_df=df, location_col=df.columns.values[0], value_col=df.columns.values[1])
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(df)
    return Status_flag


def config_options():
    st.sidebar.checkbox('Do you want to extend ROWS_PER_RESULTSET to 100?', key="extend_rows_per_resultset", value = False)


def main():
    # '--------------------------------------------------,
    # '--------------------------------------------------,
    # '--------------------------------------------------,
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


    if "messages" not in st.session_state:
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

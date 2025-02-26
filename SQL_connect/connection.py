import streamlit as st
import pandas as pd
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="Database AI Chat", layout="centered", initial_sidebar_state="collapsed")
st.title("Chat with Your Database")

# Sidebar theming and options
st.sidebar.markdown("""<style>
.sidebar .sidebar-content {
    background-color: #4b0082; 
    padding: 20px;
    border-radius: 10px;
}
</style>""", unsafe_allow_html=True)
options = ["SQLite3 (trial.db)", "MySQL"]
selected_option = st.sidebar.radio("Select Database", options)

# Database connection parameters
DB_URI = "USE_MYSQLDB" if selected_option == options[1] else "USE_LOCALDB"

def get_mysql_credentials():
    with st.sidebar.expander("MySQL Connection Details"):
        mysql_host = st.text_input("Host")
        mysql_user = st.text_input("User")
        mysql_password = st.text_input("Password", type="password")
        mysql_db = st.text_input("Database")
    return mysql_host, mysql_user, mysql_password, mysql_db

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.sidebar.warning("API key not found. Ensure GROQ_API_KEY is set.")

def chat_card(content, role):
    title_map = {"assistant": "Assistant", "user": "User"}
    title = title_map.get(role, "User")
    st.markdown(f"""
        <div style= "background-color: #A3BFB3; "border: 2px solid #333; border-radius:10px; padding:15px; margin:10px 0;">
            <h4>{title}:</h4>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)



# Configure database connection
@st.cache_resource(ttl="1h")
def configure_db(DB_URI, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if DB_URI == "USE_LOCALDB":
        dbfilepath = (Path(__file__).parent / "trial.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif DB_URI == "USE_MYSQLDB":
        if not all([mysql_host, mysql_user, mysql_password, mysql_db]):
            st.error("Incomplete MySQL details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# MySQL connection setup if chosen
if DB_URI == "USE_MYSQLDB":
    mysql_host, mysql_user, mysql_password, mysql_db = get_mysql_credentials()
    db = configure_db(DB_URI, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(DB_URI)

# Language model setup
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    memory=memory,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=10
)

# Clear history feature
if st.sidebar.button("Clear History"):
    if st.sidebar.checkbox("Confirm clear chat history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        memory.clear()
        st.sidebar.success("Chat history cleared.")
    else:
        st.sidebar.warning("Check the box to confirm clearing history.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    chat_card(msg["content"], msg["role"])

# User input and handling
user_query = st.chat_input(placeholder="Type your query here...")
if user_query:
    if user_query.lower() in ["hello", "hi", "hey", "hey there"]:
        response = "Hello! How can I assist you today?"
        # Append greeting response to session state and display
        st.session_state.messages.append({"role": "assistant", "content": response})
        chat_card(response, "assistant")
    else:
        # Append user query to session state and display
        detailed_query = f"Retrieve all names of the students. Provide the complete list without truncation."
        st.session_state.messages.append({"role": "user", "content": user_query})
        chat_card(user_query, "user")

        # The rest of your processing logic for non-greeting queries...

    # Process the input and display results
    with st.spinner("Processing your query..."):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(input=user_query, callbacks=[streamlit_callback])
            
            # Print the generated SQL query for debugging
            st.write("Generated SQL Query:", response)
            
            # Display output in a DataFrame if applicable   
            if isinstance(response, pd.DataFrame):
                st.dataframe(response)  # Display as a table
            else:
                # Ensure proper formatting of agent output
                if "Final Answer:" in response:
                    response = response.split("Final Answer:", 1)[-1].strip()
                chat_card(response.replace("\n", "<br>"), "assistant")
        except Exception as e:
            response = "I'm sorry, but I couldn't find relevant information in the database." if "parsing error" in str(e).lower() else f"An error occurred: {str(e)}"
            chat_card(response, "assistant")

# Auto-scroll to the latest message
st.markdown(
    "<script>document.getElementsByClassName('st-chat-container')[0].scrollTop = document.getElementsByClassName('st-chat-container')[0].scrollHeight;</script>",
    unsafe_allow_html=True
)

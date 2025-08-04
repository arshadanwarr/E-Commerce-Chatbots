import os
import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Load environment variables
load_dotenv()
DB_URI = os.getenv("DB_URI", "")
GROQ_API_KEY = ""

# ✅ Initialize DB and LLM once (for better performance)
@st.cache_resource
def init_db_and_llm():
    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=2
    )
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False, return_direct=True)
    return db_chain

# ✅ Prompt template
template = """
You are an AI assistant that converts natural language questions into SQL queries
and retrieves results from the database.

Question: {question}
"""
prompt = PromptTemplate(input_variables=["question"], template=template)

# ✅ Streamlit UI
st.set_page_config(page_title="SQL Chatbot", page_icon="🤖", layout="centered")
st.title("Order Tracking Bot")
st.write("Fetch any info from Database.")

db_chain = init_db_and_llm()

# ✅ Chat input
user_query = st.chat_input("Type your question here...")
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                formatted_query = prompt.format(question=user_query)
                result = db_chain.invoke({"query": formatted_query})

                # ✅ Extract raw result
                raw_result = result.get("result", "").strip("[]")

                # ✅ Try to convert to table if possible
                try:
                    # Convert tuple-like string to DataFrame
                    clean_data = eval(raw_result)  # Careful with eval; works if data is trusted
                    if isinstance(clean_data, tuple):
                        clean_data = [clean_data]
                    df = pd.DataFrame(clean_data)
                    st.write("✅ Query Result:")
                    st.dataframe(df)
                except Exception:
                    st.text(raw_result)

            except Exception as e:
                st.error(f"❌ Error: {e}")

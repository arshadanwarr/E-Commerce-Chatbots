import streamlit as st
import bs4
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.document_loaders import WebBaseLoader

# ---- Build or Load FAISS ----
def build_or_load_vectorstore(url):
    embedding = OllamaEmbeddings(model="all-minilm:22m")
    index_path = "faiss_index_new"

    try:
        vector_store = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
        print("Loaded FAISS index from disk.")
    except:
        print("Building FAISS index...")
        loader = WebBaseLoader(web_paths=[url], bs_kwargs={"parse_only": bs4.SoupStrainer("body")})
        docs = list(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, embedding)
        vector_store.save_local(index_path)
        print("Saved FAISS index to disk.")
    
    return vector_store

# ---- LLM ----
def create_llm():
    return ChatOllama(model="tinyllama:1.1b", temperature=0, max_tokens=300, streaming=True)

# ---- Prompt ----
def create_advanced_prompt():
    template = """
    You are a helpful AI assistant answering questions based on the provided context and previous conversation.
    Only return the final answer.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Helpful Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])


# ---- QA Chain ----
def create_qa_chain(llm, retriever, prompt):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ---- Streamlit App ----
st.set_page_config(page_title="Chat with Website", page_icon="🤖", layout="wide")

st.title("Chat with E-Commerce Websites")
st.write("Ask questions based on the website content!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: URL input
with st.sidebar:
    st.header("Configuration")
    url = st.text_input("Website URL", value="https://www.amazon.com")
    if st.button("Load Website"):
        with st.spinner("Loading and indexing website..."):
            st.session_state.vector_store = build_or_load_vectorstore(url)
            st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
            st.session_state.llm = create_llm()
            st.session_state.prompt = create_advanced_prompt()
            st.session_state.qa_chain = create_qa_chain(st.session_state.llm, st.session_state.retriever, st.session_state.prompt)
        st.success("Website Loaded! Start chatting below.")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (only if website loaded)
if "qa_chain" in st.session_state:
    if user_query := st.chat_input("Ask something about the website..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({"question": user_query, "chat_history": []})
                ai_response = result["answer"]
                st.markdown(ai_response)

        # Save AI response to session state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    st.info("Please enter a URL and click **Load Website** to start chatting.")

import bs4
import asyncio
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain.document_loaders import WebBaseLoader


async def load_data(url: str):
    
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": bs4.SoupStrainer("body"),  # Extract only <body>
        },
        bs_get_text_kwargs={"separator": " | ", "strip": True},
    )

    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    splits = text_splitter.split_documents(documents)
    return splits


def create_faiss_vector_store(documents, model_name="nomic-embed-text:v1.5"):
    
    embedding = OllamaEmbeddings(model=model_name)
    vector_store = FAISS.from_documents(documents, embedding)
    print(f"FAISS vector store created with {len(documents)} documents.")
    return vector_store



def create_llm(model_name="deepseek-r1:1.5b", temperature=0):
 
    return ChatOllama(model=model_name, temperature=temperature)


def create_advanced_prompt():

    template = """
    You are a helpful AI assistant answering questions based on the provided context and previous conversation.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Helpful Answer:
    """
    return PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])


def create_retriever(vector_store, k=4):

    return vector_store.as_retriever(search_kwargs={"k": k})


def create_memory():
   
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def create_conversational_chain(llm, retriever, memory, prompt):
 
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )











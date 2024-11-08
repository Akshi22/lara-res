from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import nltk

nltk.download('punkt')

load_dotenv('/.env')
API_KEY = os.getenv('API_KEY')

working_dir = os.getcwd()

TEMPERATURE = 0.3
TOP_P = 1
MAX_TOKENS_TO_GENERATE = 1024
TOP_K = 500

MODEL_ID = "llama-3.2-11b-text-preview"

def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="/n", 
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstores = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstores

def create_chain(vectorstores):
    api = API_KEY
    llm = ChatGroq(
        api_key = api,
        model = MODEL_ID,
        temperature=TEMPERATURE
    )
    
    retriever = vectorstores.as_retriever()
    
    memory = ConversationBufferMemory(
        llm = llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages = True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        verbose = True
    )
    
    return chain

st.set_page_config(page_title="🚀Resume-Assistant💰", layout='wide')
st.markdown("<h1 style='text-align: left; color:#FFEE8C'>LaRa-Res : Virtual Resume Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left;'>Hi! I'm here to boost your job prospects. Let's customize your resume together to impress recruiters.</h5>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
uploaded_file = st.file_uploader(label="Upload your Resume")
if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())       
        
    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = setup_vectorstore(load_documents(file_path)) 
        
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstores)
    
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
user_input = st.chat_input("Type your question here!")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    










import os
import tempfile
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_together import Together
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pinecone

# Initialize Pinecone
def init_pinecone(api_key, index_name):
    pinecone.init(api_key=api_key)
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Match OpenAI embedding size
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    return pinecone.Index(index_name)

# Process uploaded documents
def process_documents(uploaded_files, openai_key, pinecone_key, index_name):
    docs = []
    
    # Load PDF files
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
        os.unlink(tmp.name)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)
    
    # Create embeddings and store in Pinecone
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key
    )
    
    return PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_key
    )

# Initialize the QA chain
def init_qa_chain(vector_store, together_key):
    llm = Together(
        model="togethercomputer/llama-3-70b-chat",
        temperature=0.3,
        max_tokens=1024,
        together_api_key=together_key
    )
    
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# Streamlit UI
st.set_page_config(page_title="RAG Chat", layout="wide")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    pinecone_key = st.text_input("Pinecone API Key", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    together_key = st.text_input("Together AI API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    index_name = st.text_input("Pinecone Index Name", "rag-chat")
    
    if st.button("Initialize System"):
        if not all([pinecone_key, openai_key, together_key, uploaded_files]):
            st.error("Please provide all required credentials and documents")
        else:
            with st.spinner("Initializing..."):
                # Initialize Pinecone
                init_pinecone(pinecone_key, index_name)
                
                # Process documents
                vector_store = process_documents(
                    uploaded_files,
                    openai_key,
                    pinecone_key,
                    index_name
                )
                
                # Create QA chain
                st.session_state.qa_chain = init_qa_chain(vector_store, together_key)
                st.success("System ready!")

# Chat interface
st.title("🧠 Document Chat Assistant")
st.caption("Powered by Pinecone, Together AI, and LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about your documents"):
    if "qa_chain" not in st.session_state:
        st.error("Please initialize the system first in the sidebar")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = st.session_state.qa_chain.invoke(
                    {"question": prompt},
                    return_only_outputs=True
                )
                
                st.markdown(f"**Answer**: {response['answer']}")
                st.markdown("**Sources**:")
                for doc in response['source_documents'][:3]:
                    st.markdown(f"- `{doc.metadata['source']}` (Page {doc.metadata.get('page', 'N/A')})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer']
        })

import os
import tempfile
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_together import Together
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone connection
pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
INDEX_NAME = "ragreader"

# Validate index configuration
index_info = pc.describe_index(INDEX_NAME)
assert index_info.dimension == 1024, "Index dimension mismatch (expected 1024)"
assert index_info.metric == "cosine", "Index metric mismatch (expected cosine)"

# Initialize embeddings with explicit dimension setting
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=st.secrets.OPENAI_API_KEY,
    dimensions=1024  # Must match index dimension
)

# Document processing pipeline
def process_documents(uploaded_files):
    docs = []
    
    # Process PDF files
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
    
    # Store in Pinecone
    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_client=pc
    )

# Initialize QA chain
def init_qa_chain():
    llm = Together(
        model="togethercomputer/llama-3-70b-chat",
        temperature=0.3,
        max_tokens=1024,
        together_api_key=st.secrets.TOGETHER_API_KEY
    )
    
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_client=pc
    )
    
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# Streamlit UI
st.set_page_config(page_title="RAG Chat", layout="wide")

# Document management sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Storing documents..."):
            process_documents(uploaded_files)
            st.session_state.qa_chain = init_qa_chain()
            st.success("Documents processed!")

# Chat interface
st.title("📚 Document Chat Assistant")
st.caption(f"Connected to Pinecone index: {INDEX_NAME} (1024-dim cosine)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about your documents"):
    if "qa_chain" not in st.session_state:
        st.error("Please upload and process documents first")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = st.session_state.qa_chain.invoke(
                    {"question": prompt},
                    return_only_outputs=True
                )
                
                st.markdown(f"**Answer**: {response['answer']}")
                st.markdown("**Relevant Sources**:")
                for doc in response['source_documents'][:3]:
                    source = os.path.basename(doc.metadata['source'])
                    page = doc.metadata.get('page', 'N/A')
                    st.markdown(f"- `{source}` (Page {page})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer']
        })

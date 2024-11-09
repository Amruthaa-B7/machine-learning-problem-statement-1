import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"]
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text
  def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=8000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
  def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
def get_query_generation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    memory = ConversationBufferMemory(return_messages=True)
    retriever = vectorstore.as_retriever()
    
    # Template for generating Boolean queries
    template = """You are an AI assistant that generates Boolean search queries based on context.
    
    Context: {context}
    Query Request: {question}
    
    Generate a Boolean search query for the given context and question."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain, memory
def handle_userinput(user_request):
    if st.session_state.query_chain is None:
        st.error("Please upload and process a PDF document first!")
        return

    try:
        response = st.session_state.query_chain.invoke(user_request)
        
        # Display the generated Boolean query in a scrollable area
        with st.expander("Generated Boolean Queries", expanded=True):
            for user_query, generated_query in st.session_state.query_history:
                st.write(f"User Query: **{user_query}**")
                st.write(f"Boolean Query: `{generated_query}`")

        # Store history
        st.session_state.query_history.append((user_request, response))

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="AI-Driven Boolean Query Generator", page_icon=":mag:")
    
    # Initialize session state
    if "query_chain" not in st.session_state:
        st.session_state.query_chain = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = False

    # App Header
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>AI-Driven Boolean Query Generator</h1>", unsafe_allow_html=True)

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Choose PDFs", accept_multiple_files=True
        )
        
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
                return
                
            with st.spinner("Processing your documents..."):
                try:
                    # Reset query chain and history
                    st.session_state.query_chain = None
                    st.session_state.query_history = []
                    
                    # Process PDF files
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # Initialize query generation chain with memory
                    st.session_state.query_chain, memory = get_query_generation_chain(vectorstore)
                    st.session_state.processed_pdfs = True
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
                    st.session_state.processed_pdfs = False

    # Input box for user query
    if st.session_state.processed_pdfs:
        user_request = st.text_input("Enter a search context or question:")
        if user_request:
            handle_userinput(user_request)
    else:
        st.info("Please upload and process PDF documents to start generating queries!")

if __name__ == '__main__':
    main()


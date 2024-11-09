import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAQXWNpCHehBMou_hb26CpVuwaOWLsVN1s"
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Create template
    template = """Your a Ai assistant that answer about medical questions. Use the following context to answer the question.
    
    Context: {context}
    
    Question: {question}
    
    Answer in a helpful and detailed way."""
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process a PDF document first!")
        return

    try:
        # Invoke the chain
        response = st.session_state.conversation.invoke(user_question)
        
        # Add to chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=response)
        ])

        # Display messages
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = False

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload PDF documents first!")
                return
                
            with st.spinner("Processing"):
                try:
                    # Reset the conversation
                    st.session_state.conversation = None
                    st.session_state.chat_history = []
                    
                    # Process PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.processed_pdfs = True
                    
                    st.success("PDFs processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
                    st.session_state.processed_pdfs = False

    if st.session_state.processed_pdfs:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.info("Please upload and process PDF documents to start chatting!")


if __name__ == '__main__':
    main()

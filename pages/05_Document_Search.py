from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os
import streamlit as st
from io import StringIO
import utils



def file_selector():
        folder_path = os.path.join(".", "data", "pdfs")
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        
        return os.path.join(folder_path, selected_filename), selected_filename



if __name__ == "__main__":
    
    utils.set_openai_api_key()

    st.title("AI Document Search")

    filename, selected_doc = file_selector()
    st.write('You selected `%s`' % filename)

    if filename is not None:
        abs_path= os.path.abspath(filename)

        try:
             selected_db = FAISS.load_local(os.path.join(".", "data", "db", "document_search_folder", selected_doc), HuggingFaceEmbeddings())
        except:
             print("Error with finding DB")
             with st.spinner("Making DB..."):
                raw_documents = PyPDFLoader(abs_path).load_and_split()
                
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                documents = text_splitter.split_documents(raw_documents)

                selected_db = FAISS.from_documents(documents, HuggingFaceEmbeddings())
                selected_db.save_local(os.path.join(".", "data", "db", "document_search_folder", selected_doc))

        question = st.text_input("Enter your question about this file.")
        if question:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.session_state.openai_api_key)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=selected_db.as_retriever())
            with st.spinner("Searching..."):
                res = qa_chain({"query": question})
            st.write(res["result"])
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from streamlit_js_eval import streamlit_js_eval

import os
import streamlit as st
from io import StringIO
import utils


def clear():
    st.session_state["text"] = ""


def file_selector():
        folder_path = os.path.join(".", "data", "db", "document_search_folder")
        try:
            filenames = os.listdir(folder_path)
        except:
            os.mkdir(folder_path)
            filenames = os.listdir(folder_path)
        
        filenames.append("Upload a file")
        selected_filename = st.selectbox('Select a file', filenames, on_change=clear)
        
        return os.path.join(folder_path, selected_filename), selected_filename



def file_upload():
    uploadedfile = st.file_uploader("Choose a file", "pdf")
    if uploadedfile is not None:
        texts = ""
        
        file = PdfReader(uploadedfile)
        with st.spinner("Loading pdf..."):
            for page in file.pages:
                texts += page.extract_text()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            doc = splitter.create_documents([texts])

            selected_db = FAISS.from_documents(doc, HuggingFaceEmbeddings())
            selected_db.save_local(os.path.join(".", "data", "db", "document_search_folder", uploadedfile.name))
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        return uploadedfile.name


#when i upload a file make the multiselect automatically switch to the new file



if __name__ == "__main__":
    
    utils.set_openai_api_key()

    st.title("AI Document Search")

    filename, selected_doc = file_selector()

    if filename:
        if selected_doc == "Upload a file":
             selected_doc = file_upload()
        else:
             st.write('You selected `%s`' % filename)
             abs_path= os.path.abspath(filename)
             selected_db = FAISS.load_local(os.path.join(".", "data", "db", "document_search_folder", selected_doc), HuggingFaceEmbeddings())
             question = st.text_input("Enter your question about this file.", key="text")
             if question:
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.session_state.openai_api_key)
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=selected_db.as_retriever())
                with st.spinner("Searching..."):
                    res = qa_chain({"query": question})
                st.write(res["result"])
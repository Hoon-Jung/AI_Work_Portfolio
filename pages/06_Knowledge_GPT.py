from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from streamlit_js_eval import streamlit_js_eval
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.messages import AIMessage, HumanMessage

import os
import streamlit as st
import utils

from langchain.chat_models import ChatOpenAI

from langchain.callbacks.base import BaseCallbackHandler

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder
)



class textStreamingHandler(BaseCallbackHandler):
    def __init__(self, stcontainer):
        self.tb = stcontainer
        self.text = ""
    def on_llm_new_token(self, token,**kwargs):
        self.text += token
        self.tb.write(self.text)


@st.cache_resource
def config_openai(db_name):
    selected_db = FAISS.load_local(os.path.join(".", "data", "db", "document_search_folder", db_name), HuggingFaceEmbeddings())
    
    retriever = selected_db.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    condense_q_system_prompt = """Given a chat history and the latest user question \
    which might reference the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    condense_q_chain = condense_q_prompt | llm | StrOutputParser()


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | qa_prompt
        | llm
    )

    return rag_chain


def clear():
    st.session_state["text"] = ""
    st.session_state.msg = []


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



if __name__ == "__main__":
    
    utils.set_openai_api_key()

    st.title("AI Document Search")

    filename, selected_doc = file_selector()
    chat_history = []

    if filename:
        if selected_doc == "Upload a file":
             selected_doc = file_upload()
        else:
             st.write('You selected `%s`' % filename)

             if "msg" not in st.session_state:
                st.session_state.msg = []
            
             abs_path= os.path.abspath(filename)
             qa_chain = config_openai(selected_doc)
        
             for message in st.session_state.msg:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

             prompt = st.chat_input("say something")
             if prompt:
                with st.chat_message("user"):
                    st.write(prompt)
                    st.session_state.msg.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    textbox = textStreamingHandler(st.empty())
                    ai_msg = qa_chain.invoke({"question": prompt, "chat_history": chat_history}, config={"callbacks": [textbox]})
                    chat_history.extend([HumanMessage(content=prompt), ai_msg])
                    st.session_state.msg.append({"role": "assistant", "content": ai_msg.content})
             
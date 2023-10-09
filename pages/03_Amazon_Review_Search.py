from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate

import openai
import streamlit as st
import os
import pandas as pd
import json


# @st.cache_data
def load_data():
    data_path = "./data/AMAZON_FASHION_5.json"
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return df


def prep_db():

    df = load_data()
    products = df["asin"].unique()
    for name in products:
        same_products = df[df["asin"]==name]["reviewText"]
        reviews_doc = same_products.tolist()
        reviews_doc = "\n".join(reviews_doc)

        text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000)
        texts = text_splitter.create_documents([reviews_doc])

        db = FAISS.from_documents(texts, HuggingFaceEmbeddings())

        db.save_local(f"./data/db/{name}")



def get_answer(product_id, question, apikey):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=apikey)

    load_db = FAISS.load_local(f"./data/db/{product_id}", HuggingFaceEmbeddings())

    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=load_db.as_retriever())
    res = qa_chain({"query": question})

    return res["result"]

if __name__ == "__main__":
    st.title("Review Search Engine")
    options = ["B00007GDFV", "B00008JOQI", "7106116521"]
    api_key_input = st.text_input("Enter OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")

    if not os.path.exists("./data/db/B00007GDFV/index.pkl"):
        prep_db()
        
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        selected_option = st.selectbox("Select a product", options)
        st.write("Currently selected:", selected_option)
        question = st.text_input("Ask a question")
        if st.button("Get Answer"):
            st.write(get_answer(selected_option, question, openai_api_key))
    else:
        st.warning("WARNING: Enter your OpenAI API key!")
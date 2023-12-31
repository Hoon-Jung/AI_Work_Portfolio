import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain import LLMChain
from tqdm import tqdm
import os
import ast
import streamlit as st
import openai

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate

)

import utils



@st.cache_data
def load_data():

    data_path = "./data/AMAZON_FASHION_5.json"

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    
    return df


def make_tags(review):

    chat = ChatOpenAI(openai_api_key=st.session_state.openai_api_key)

    template = "A tagging system that creates tags for use in an online shopping mall."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Create up to 5 tags for the given review. The result should be a python style array of strings. And it should only be formated like this [..., ..., ..., ..., ...] and if it is not possible to make 5 tags then create the highest possible number of tags:  '''{text}'''"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    
    chainai = LLMChain(llm=chat, prompt=chat_prompt)
    result = chainai.run(text=review)
    return result


def str_to_list(strings):
    try:
        parsed_list = ast.literal_eval(strings)
        return parsed_list
    except Exception as e:
        print(strings)
        print("Error parsing", e)
        return []


def file_loaded(file_name):
    review_tags = pd.read_csv(file_name)
    tag_column_df = review_tags['tags'].apply(str_to_list)
    all_tags = {}
    for tags in tag_column_df:
        for tag in tags:
            all_tags[tag] = all_tags.get(tag, 0) + 1
    all_sorted = dict(sorted(all_tags.items(), key=lambda item: item[1]))
    top_10 = []
    for i in range(-1, -11, -1):
        top_10.append(list(all_sorted)[i])
    return top_10


def load_all_reviews(product, df):
    same_products = df[df["tags"].apply(lambda x: product in x)]
    for i,info in same_products.iterrows():
        st.write(f"**Overall Score:** {info['overall']}")
        st.write(f"**Product ID:** {info['asin']}")
        st.write(f"**Reviewer Name:** {info['reviewerName']}")
        st.write(f"**Review Text:**")
        st.write(f"'{info['reviewText']}'")
        st.write(f"**Tags:** {info['tags']}")
        st.markdown('<hr>', unsafe_allow_html=True)


def generate_file(df, file_name):
    with st.spinner("Making tags..."):
        df['tags'] = df.apply(lambda x: make_tags(x["reviewText"]), axis=1)
    df.to_csv(file_name, index=False)



if __name__ == "__main__":
    utils.set_openai_api_key()

    st.title("Amazon Review Tagging")

    df = load_data()
    file_path = "./data/AMAZON_FASHION_TAGS_50.csv"
    text = st.empty()
    if not os.path.exists(file_path):
        generate_file(df, file_path)

    highest_10 = file_loaded(file_path)
   
    selected_option = st.selectbox("Select a tag", highest_10)
    st.write("Currently selected:", selected_option)
    get_all_tags_btn = st.button("Get Tags")
    if get_all_tags_btn:
        load_all_reviews(selected_option, pd.read_csv(file_path))
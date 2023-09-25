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

@st.cache_data
def load_data():

    data_path = "./data/AMAZON_FASHION_5.json"

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    
    return df


def make_tags(review, key):

    chat = ChatOpenAI(openai_api_key=key)

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


if __name__ == "__main__":
    api_key_input = st.text_input("Enter OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")
    make_tag_button = st.button("Make Tags")
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
        df = load_data()
        file_path = "./data/AMAZON_FASHION_TAGS.csv"
        text = st.empty()
        if make_tag_button:
            if not os.path.exists(file_path):
                text.text("File does not exist")
                with st.spinner("Making tags..."):
                    df['tags'] = df.apply(lambda x: make_tags(x["reviewText"], openai_api_key), axis=1)
                df.to_csv(file_path, index=False)
                text.text("File exists")
                st.write(pd.read_csv(file_path))
            else:
                text.text("File exists")
                st.write(pd.read_csv(file_path))
    
    else:
        st.warning("WARNING: Enter your OpenAI API key!")

    
    # tqdm.pandas()
    





    # review_tags = pd.read_csv("./AMAZON_FASHION.json/OFFICIAL_AMAZON_FASHION_TAGS.csv")
    # all_tags={}
    # tag_column_df = review_tags['tags'].apply(str_to_list)
    # # print(tag_column_df)
    # for tags in tag_column_df:
    #     for tag in tags:
    #         all_tags[tag] = all_tags.get(tag, 0) + 1
    # # print(all_tags)
    # print(dict(sorted(all_tags.items(), key=lambda item: item[1])))

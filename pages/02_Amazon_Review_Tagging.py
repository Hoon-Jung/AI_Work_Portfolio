import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain import LLMChain
from tqdm import tqdm
import os
import ast

import streamlit as st

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate

)


def load_data():

    data_path = "./AMAZON_FASHION.json/AMAZON_FASHION.json"

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    
    return df


def make_tags(review):

    chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

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
    df = load_data()
    st.write(df)
    # tqdm.pandas()
    # df['tags'] = df.progress_apply(lambda x: make_tags(x["reviewText"]), axis=1)
    # df.to_csv("./AMAZON_FASHION.json/OFFICIAL_AMAZON_FASHION_TAGS.csv", index=False)





    # review_tags = pd.read_csv("./AMAZON_FASHION.json/OFFICIAL_AMAZON_FASHION_TAGS.csv")
    # all_tags={}
    # tag_column_df = review_tags['tags'].apply(str_to_list)
    # # print(tag_column_df)
    # for tags in tag_column_df:
    #     for tag in tags:
    #         all_tags[tag] = all_tags.get(tag, 0) + 1
    # # print(all_tags)
    # print(dict(sorted(all_tags.items(), key=lambda item: item[1])))

import streamlit as st

import openai
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
import utils

import os
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


def config_openai(p):
    ai = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.session_state.openai_api_key)

    template = "A Friendly assistant"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "You will respond to this chat in a friendly manner: {text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    # chainai = LLMChain(llm=ai, prompt=chat_prompt)

    # prompt_template = "Respectfully answer: {chat}?"
    chainai = LLMChain(llm = ai, prompt=chat_prompt)
    result = chainai(p)

    return result["text"]



if __name__ == "__main__":
    utils.set_openai_api_key()
    
    if "msg" not in st.session_state:
        st.session_state.msg = []

    for message in st.session_state.msg:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    prompt = st.chat_input("say something")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.msg.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
             answer = config_openai(prompt)
             st.write(answer)
             st.session_state.msg.append({"role": "assistant", "content": answer})


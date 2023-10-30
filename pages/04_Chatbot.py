import streamlit as st

import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import utils

from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
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
def config_openai():
    ai = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.session_state.openai_api_key, streaming=True)

    template = "A Friendly assistant"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "You will respond to this chat in a friendly manner: {text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, MessagesPlaceholder(variable_name="chat_history"),human_message_prompt])
    
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    chainai = LLMChain(llm = ai, prompt=chat_prompt, memory=memory)

    return chainai



if __name__ == "__main__":
    utils.set_openai_api_key()
    st.title("Simple AI Chatbot")
    ai = config_openai()
    
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
             textbox = textStreamingHandler(st.empty())
             answer = ai.run(text=prompt, callbacks=[textbox])
             st.session_state.msg.append({"role": "assistant", "content": answer})


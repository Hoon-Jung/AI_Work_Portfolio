import streamlit as st
import openai
import os

import utils



if __name__ == "__main__":
    st.set_page_config(page_title="Product Description Generator")
    st.title("Product Description Generator")
    aidesc = ""
    st.write(st.session_state)
    utils.set_openai_api_key()
    st.write("utils went")
    st.write(st.session_state)
    desc = st.text_input("Product Information", placeholder="Enter a short product description")
    aidescription = st.empty()

    openai.api_key = st.session_state.openai_api_key

    if st.button("Generate Product Description"):
        response = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",
            messages = [{"role": "system", "content": "write a product description in 5 sentences for the given product"}, {"role": "user", "content": desc}],
            stream=True
        )
            
            
        for chunk in response:
            if "content" in chunk.choices[0]["delta"]:
                aidesc = aidesc + chunk.choices[0]["delta"]["content"]
                aidescription.empty()
                aidescription.write(aidesc)


    
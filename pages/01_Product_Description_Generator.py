import streamlit as st
import openai
import os



if __name__ == "__main__":
    st.set_page_config(page_title="Product Description Generator")
    st.title("Product Description Generator")
    aidesc = ""
    api_key_input = st.text_input("Enter OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")
    desc = st.text_input("Product Information", placeholder="Enter a short product description")
    aidescription = st.empty()

    
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key

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

    else:
        st.warning("WARNING: Enter your OpenAI API key!")


    
import streamlit as st
import json
import openai
import os

openai.organization = "org-BKMBE8WdTAJKgRdZfV9tMxBt"


if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = os.environ['OPENAI_API_KEY']
    st.sidebar.write(openai.api_key)
else:
    openai.api_key = st.sidebar.text_input("key")

if openai.api_key != "":
    models = openai.Model.list()

prompt = st.text_area("Enter your input here")
model = st.sidebar.selectbox("Select engine", ['text-davinci-003', 'text-davinci-002'])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
response_length = st.sidebar.slider("Response length", 0, 1000, 200, 10)

if st.button("Run"):
    response = openai.Completion.create(
        model = model,
        prompt = prompt,        
        max_tokens = response_length,
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    st.write(response['choices'][0]['text'])


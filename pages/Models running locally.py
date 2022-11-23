import streamlit as st
from transformers import GPTJForCausalLM, AutoTokenizer
import torch


# Load model and tokenizer
#@st.cache(allow_output_mutation=True)
def load_model():
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

load_model()

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
min_length = st.sidebar.slider("Min length", 0, 100, 100, 10)
max_length = st.sidebar.slider("Max length", 0, 1000, 200, 10)

st.header("GPT playgrond running locally")
prompt = st.text_area("Enter your prompt here:")

if st.button("Run"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    
    data = {
    'text': prompt,
    'parameters:': {
        'min_length': min_length,
        'max_length': max_length,
        'temperature': temp,
        }
    }

    st.write(gen_text)

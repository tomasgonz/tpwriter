import streamlit as st
from transformers import pipeline
gen = pipeline('text-generation', model ='EleutherAI/gpt-neo-2.7B')
#gen = pipeline('text-generation', model ='EleutherAI/gpt-neo-1.3B')
#gen = pipeline('text-generation', model ='EleutherAI/gpt-neo-125M')
context = st.text_input('Context')

if context != '':
    output = gen(context, max_length=200, do_sample=True, temperature=0.9)
    with open('dl.txt', 'w') as f:
        f.write(str(output))

    st.write(output[0]['generated_text'])
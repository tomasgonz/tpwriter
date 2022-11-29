import streamlit as st
import json
import openai
import os

openai.organization = "org-BKMBE8WdTAJKgRdZfV9tMxBt"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

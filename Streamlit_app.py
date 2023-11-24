import os
import streamlit as st
from htmlTemplates import css, bot_template, user_template


#os.environ["OPENAI_API_KEY"] = "abc"
openai_api_key = os.environ.get("OPENAI_API_KEY")
PASSWORD = "sgh"

st.write('Hello world! testing 1 2 3 ')

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import tempfile
from model import Retrieval_QA

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title='🦜🔗 Quick GPT')
st.title("Quick GPT 📖📚🏫📝")
st.sidebar.title("API Key")
st.sidebar.subheader("Steps: ")
st.sidebar.markdown("1 - Input your OpenAI API key")
st.sidebar.markdown("2 - Insert a pdf file")
st.sidebar.markdown("3 - Asks questions about your file")
st.sidebar.markdown("4 - Have fun 😊")

Open_AI_key = st.sidebar.text_input('Key:', type = 'password', value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""))

st.session_state["OPENAI_API_KEY"] = Open_AI_key

pdf = st.file_uploader("Please insert your document here in PDF format: ", type="pdf")

temp_file_path = os.getcwd()

if pdf is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, pdf.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(pdf.read())

        st.write("Full path of the uploaded file:", temp_file_path)

query_input = st.text_area(f"What do you want to know about?", disabled=not Open_AI_key and not pdf)
submitted = st.button('Submit')

with st.spinner("Thinking about it..."):
    if submitted and Open_AI_key.startswith('sk-'):
        result = Retrieval_QA(temp_file_path, query_input, Open_AI_key)
    elif submitted and not Open_AI_key.startswith("sk-"):
        st.warning('Please enter a valid OpenAI API key!', icon='⚠')
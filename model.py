import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.docarray.in_memory import DocArrayInMemorySearch

def Retrieval_QA(pdf, query_input, key):

    # Load and split the desired document
    loader = PyPDFLoader(pdf)
    
    texts = loader.load_and_split()

    # Define the model, embedding and vectorstore
    llm = ChatOpenAI(streaming=True, temperature=0.0, max_tokens=1000, openai_api_key=key, callbacks=[StreamingStdOutCallbackHandler()])
    embeddings = OpenAIEmbeddings(openai_api_key=key)

    db = Chroma.from_documents(
        texts,
        embeddings
    )

    # Define retriever and memory variable to use in the model
    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Define the model and run the query
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    answer = qa({"question": query_input}, return_only_outputs=True)

    return st.write(answer)
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores.docarray.in_memory import DocArrayInMemorySearch

def Retrieval_QA(pdf, query_input, key):

    """
    This function retrieves answers to questions from a PDF document using a conversational model.

    Parameters:
        pdf (str): The path to the PDF document.
        query_input (str): The question or query for which you want to find an answer.
        key (str): Your OpenAI API key for language model and embeddings.

    Returns:
        answer: The answer to the user's question is displayed using Streamlit.

    Workflow:
    1. Load and split the PDF document into text segments.
    2. Initialize the language model (ChatOpenAI) and word embeddings (OpenAIEmbeddings) using the provided API key.
    3. Create a Chroma vector store from the document texts.
    4. Define a retriever to search for relevant information in the document.
    5. Set up a memory buffer to maintain a conversation history.
    6. Create a ConversationalRetrievalChain using the language model, retriever, and memory.
    7. Query the model with the user-provided question.
    8. Display the answer using Streamlit.
    """

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
***Quick GPT*** 📖📚🏫📝

Quick GPT is a Streamlit-based application that allows users to upload PDF documents and ask questions about their content using OpenAI's GPT model. Features - Upload PDF documents- Ask questions about the uploaded document - Get AI-generated answers based on the document content - User-friendly interface with Streamlit. Requirements Streamlit, LangChain, OpenAI API key. Installation: Clone this repository: git clone https://github.com/EdgarSilva-tech/Streamlit-PDF-App.git and change into this directory as so cd Streamlit-PDF-App. Install the required packages: pip install -r requirements.txt. Set up your OpenAI API key: Create a `.env` file in the project root - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`. 

Usage: Run the Streamlit app: streamlit run main.py. Open your web browser and go to the provided local URL (usually `http://localhost:8501`). Enter your OpenAI API key in the sidebar (if not already set in the `.env` file). Upload a PDF document and ask questions about the document in the text area. Click "Submit" to get AI-generated answers. How it works: The app uses LangChain to process the uploaded PDF document. It creates embeddings of the document content using OpenAI's embedding model. The embeddings are stored in a Chroma vector store for efficient retrieval. When a question is asked, the app uses a ConversationalRetrievalChain to find relevant information and generate an answer. The answer is displayed to the user using Streamlit.

Files: `main.py`: The main Streamlit application\n- `model.py`: Contains the `Retrieval_QA` function for processing documents and generating answers, `requirements.txt`: List of required Python packages 

![image](https://github.com/EdgarSilva-tech/Streamlit-PDF-App/assets/81367614/2528b7f4-e1fd-4734-a2c2-c1cff4d6d485)


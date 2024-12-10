from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain_core.prompts import ChatPromptTemplate  
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Llama 3.1 model locally using OllamaLLM
llm = OllamaLLM(model="llama3.1", local_path="/models/llama-2-7b-chat.ggmlv3.q4_0.bin")

# Create a custom prompt template for medical queries
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI medical assistant. 
    Use the following context from medical documents to provide accurate and helpful information. 
    If the context doesn't contain enough information, clearly state that you can only provide limited information based on the available context.
    Always prioritize patient safety and recommend consulting a healthcare professional for definitive medical advice."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a comprehensive and clear answer based on the context."),
])

# Create the RAG chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        msg = request.form["msg"]
        logging.debug(f"Received message: {msg}")
        
        # Create input data for RAG chain
        input_data = {"input": msg}
        logging.debug(f"Input data for RAG chain: {input_data}")
        
        # Invoke the RAG chain directly
        response = rag_chain.invoke(input_data)
        
        logging.debug(f"Generated response: {response}")
        return str(response["answer"])
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        return "Sorry, I encountered an error processing your request."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from dotenv import load_dotenv
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Pinecone Vector Store
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not found in environment variables.")
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
except Exception as e:
    logger.error(f"Error initializing PineconeVectorStore: {e}")
    raise e

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Groq API key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY not found in environment variables.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# System prompt as part of the chat context
SYSTEM_PROMPT = """You are an AI medical assistant. 
Use the following context from medical documents to provide accurate and helpful information. 
If the context doesn't contain enough information, clearly state that you can only provide limited information based on the available context.
Always prioritize patient safety and recommend consulting a healthcare professional for definitive medical advice."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        msg = request.form.get("msg", "")
        if not msg:
            logger.warning("No message received in the request.")
            return "No message received.", 400

        logger.debug(f"Received message: {msg}")

        # Get relevant documents from Pinecone
        docs = retriever.get_relevant_documents(msg)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create messages array for Groq
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {msg}\n\nProvide a comprehensive and clear answer based on the context, and dont give medical answer answer if not asked for and if any normal question is asked then reply accordingly"}
        ]

        # Get response from Groq
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=200,
            )
            response = chat_completion.choices[0].message.content
            logger.debug(f"Response from model: {response}")
            return str(response)

        except Exception as e:
            logger.error(f"Error in Groq API call: {e}")
            raise e

    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return "Sorry, I encountered an error processing your request.", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
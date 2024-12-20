from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

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

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY not found in environment variables.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)

CLASSIFICATION_PROMPT = """
You are an intelligent assistant tasked with classifying user queries. 
Your job is to decide if a query is:
1. A "Medical Query" related to medical issues, symptoms, or treatments.
2. A "General Query" that is conversational or unrelated to medical topics.
3. An "Irrelevant Query" that is outside your scope.

Return one label: "Medical Query", "General Query", or "Irrelevant Query".

Query: {query}
"""

GENERAL_PROMPT = """
You are a medical assistant chatbot. Your role is to provide medical-related information and assistance. 
You should only respond to general queries like greetings ("hello", "hi") or polite conversational phrases ("how are you?", "thank you").

For any other general query or non-medical question (e.g., 'What is Ferrari?', 'Who is the president?'), politely inform the user:
- "I am here to assist with medical-related questions only. Please ask me something related to health or medicine."

Examples:
- Input: "Hello"
  Response: "Hello! How can I assist you with medical-related questions today?"
- Input: "How are you?"
  Response: "I'm doing well, thank you for asking! How can I assist with your medical concerns today?"
- Input: "What is Ferrari?"
  Response: "I am here to help with medical-related questions only. Please let me know if you have any health concerns."
"""


IRRELEVANT_PROMPT = """
You are a conversational AI assistant. For irrelevant or out-of-scope queries, politely let the user know you cannot assist with their request.
"""

FINAL_RESPONSE_PROMPT = """
You are an AI medical assistant. Based on the context provided below, create a clear, well-structured, and comprehensive response to the user's queryand note that it should be meaningfull and not like its not making sence according to the question asked. 
Always prioritize user safety and recommend consulting a healthcare professional when necessary.

Context:
{context}

Query:
{query}
"""


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

        classification_message = [{"role": "system", "content": CLASSIFICATION_PROMPT.format(query=msg)}]
        try:
            classification_response = groq_client.chat.completions.create(
                messages=classification_message,
                model="mixtral-8x7b-32768",
                temperature=0.0,
                max_tokens=10
            )
            raw_classification_result = classification_response.choices[0].message.content.strip()
            classification_result = raw_classification_result.strip('"').strip("'").lower().strip()
            logger.debug(f"Raw classification result: {raw_classification_result}")
            logger.debug(f"Normalized classification result: {classification_result}")
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return "Sorry, I encountered an error processing your request.", 500

        if any(x in classification_result for x in ["medical", "medical query"]):
            classification_result = "medical query"
        elif any(x in classification_result for x in ["general", "general query"]):
            classification_result = "general query"
        elif any(x in classification_result for x in ["irrelevant", "irrelevant query"]):
            classification_result = "irrelevant query"

        if classification_result == "general query":
            print("Query classified as: General Query")  
            general_message = [
                {"role": "system", "content": GENERAL_PROMPT},
                {"role": "user", "content": f"Query: {msg}"}
            ]
            try:
                general_response = groq_client.chat.completions.create(
                    messages=general_message,
                    model="mixtral-8x7b-32768",
                    temperature=0.7,
                    max_tokens=200
                )
                response = general_response.choices[0].message.content
                logger.debug(f"General query response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error in handling general query: {e}")
                return "Sorry, I encountered an error processing your request.", 500

        elif classification_result == "irrelevant query":
            print("Query classified as: Irrelevant Query")  
            irrelevant_message = [
                {"role": "system", "content": IRRELEVANT_PROMPT},
                {"role": "user", "content": f"Query: {msg}"}
            ]
            try:
                irrelevant_response = groq_client.chat.completions.create(
                    messages=irrelevant_message,
                    model="mixtral-8x7b-32768",
                    temperature=0.7,
                    max_tokens=150
                )
                response = irrelevant_response.choices[0].message.content
                logger.debug(f"Irrelevant query response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error in handling irrelevant query: {e}")
                return "Sorry, I encountered an error processing your request.", 500

        elif classification_result == "medical query":
            print("Query classified as: Medical Query")  
            
            # Use original msg directly
            docs = retriever.get_relevant_documents(msg)
            context = "\n".join([doc.page_content for doc in docs])

            final_response_message = [
                {"role": "system", "content": FINAL_RESPONSE_PROMPT.format(context=context, query=msg)}
            ]
            try:
                final_response = groq_client.chat.completions.create(
                    messages=final_response_message,
                    model="mixtral-8x7b-32768",
                    temperature=0.7,
                    max_tokens=300
                )
                response = final_response.choices[0].message.content
                logger.debug(f"Final medical query response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error in generating final medical query response: {e}")
                return "Sorry, I encountered an error processing your request.", 500

        else:
            print("Query classified as: Unexpected Classification") 
            logger.warning(f"Unexpected classification result: {classification_result}")
            return "Sorry, I couldn't classify your query. Please try again.", 400

    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return "Sorry, I encountered an error processing your request.", 500

@app.route('/process-voice', methods=['POST'])
def process_voice():
    try:
        voice_text = request.json.get('text', '')
        if not voice_text:
            logger.warning("No voice text received in the request.")
            return "No voice input received.", 400

        # Use the existing chat logic
        logger.debug(f"Received voice text: {voice_text}")
        
        # Simulate a form request by creating a custom Request object
        with app.test_request_context('/chat', method='POST', data={'msg': voice_text}):
            return chat()

    except Exception as e:
        logger.error(f"Error in voice processing: {e}")
        return "Sorry, I encountered an error processing your voice input.", 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

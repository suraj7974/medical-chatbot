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

# Prompts for classification, general queries, and medical processing
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
You are a conversational AI assistant. Respond appropriately to general queries in a friendly and informative manner.
"""

IRRELEVANT_PROMPT = """
You are a conversational AI assistant. For irrelevant or out-of-scope queries, politely let the user know you cannot assist with their request.
"""

STRUCTURE_PROMPT = """
You are a helpful AI assistant. Structure the following medical query into a clear, concise format suitable for retrieving relevant context.

Query: {query}
"""

FINAL_RESPONSE_PROMPT = """
You are an AI medical assistant. Based on the context provided below, create a clear, well-structured, and comprehensive response to the user's query. 
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

        # Step 1: Classify the query
        classification_message = [{"role": "system", "content": CLASSIFICATION_PROMPT.format(query=msg)}]
        try:
            classification_response = groq_client.chat.completions.create(
                messages=classification_message,
                model="mixtral-8x7b-32768",
                temperature=0.0,
                max_tokens=10
            )
            # Normalize classification result
            raw_classification_result = classification_response.choices[0].message.content.strip()
            classification_result = raw_classification_result.strip('"').lower()
            logger.debug(f"Raw classification result: {raw_classification_result}")
            logger.debug(f"Normalized classification result: {classification_result}")
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return "Sorry, I encountered an error processing your request.", 500

        # Step 2: Handle General Query
        if classification_result == "general query":
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
            # Step 4.1: Structure the query
            structure_message = [
                {"role": "system", "content": STRUCTURE_PROMPT.format(query=msg)}
            ]
            try:
                structured_response = groq_client.chat.completions.create(
                    messages=structure_message,
                    model="mixtral-8x7b-32768",
                    temperature=0.5,
                    max_tokens=50
                )
                structured_query = structured_response.choices[0].message.content.strip()
                logger.debug(f"Structured query: {structured_query}")
            except Exception as e:
                logger.error(f"Error in structuring query: {e}")
                return "Sorry, I encountered an error processing your request.", 500

            # Step 4.2: Retrieve context from Pinecone
            docs = retriever.get_relevant_documents(structured_query)
            context = "\n".join([doc.page_content for doc in docs])

            # Step 4.3: Generate final response using context
            final_response_message = [
                {"role": "system", "content": FINAL_RESPONSE_PROMPT.format(context=context, query=structured_query)}
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
            # Step 5: Fallback response for unexpected classification result
            logger.warning(f"Unexpected classification result: {classification_result}")
            return "Sorry, I couldn't classify your query. Please try again.", 400

    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return "Sorry, I encountered an error processing your request.", 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

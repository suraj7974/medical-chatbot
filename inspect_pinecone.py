from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

# Initialize Pinecone
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Test query to fetch documents
query = "what is acne"
retrieved_docs = retriever.get_relevant_documents(query)

print("Retrieved Documents Structure:")
for doc in retrieved_docs:
    print(doc)

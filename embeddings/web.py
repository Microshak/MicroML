# Base Python data handling environment imports 
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

from langchain.chains.conversation.memory \
import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter



model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from os import environ

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"



loader = WebBaseLoader([
    "https://www.federalregister.gov/documents/2023/09/21/2023-20476/truth-in-lending-regulation-z-annual-threshold-adjustments-credit-cards-hoepa-and-qualified",
])

docs = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=14048, chunk_overlap=0)
docs = text_splitter.split_documents(docs)


# Set up a vector store used to save the vector embeddings. Here we use Milvus as the vector store.
vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
)


query = "What are people saying about democrats?"
doc = vector_store.similarity_search(query)

print(doc)
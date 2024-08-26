import os
from dotenv import load_dotenv

load_dotenv()
LANGCHAIN_TRACING_V2="true"
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()
search.invoke("what is the weather in SF")


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()




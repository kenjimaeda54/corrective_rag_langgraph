import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter =  RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,chunk_overlap=0
)
docs_split = text_splitter.split_documents(docs_list)

# chroma_db = Chroma.from_documents(
#     collection_name="rag_chroma",
#     documents=docs_split,
#     embedding=embeddings,
#     persist_directory="./.chroma_db"
# )

retriever = Chroma(
    collection_name="rag_chroma",
    persist_directory="./.chroma_db",
    embedding_function=embeddings
).as_retriever()



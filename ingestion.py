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

#docs_list e preciso porque o docs acima se torna um array de documentos ja que possuimos varias urls
docs_list = [doc for sublist in docs for doc in sublist]


#Usar o from_tiktoken esta medindo o tamanho dos chunks em tokens, e nao em caracteres.
#Mede em tokens, contados pelo tiktoken. Cada chunk tem no máximo 200 tokens, sem repetição entre chunks.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=0
)
docs_split = text_splitter.split_documents(docs_list)

chroma_db = Chroma.from_documents(
    collection_name="chroma_rag",
    documents=docs_split,
    embedding=embeddings,
    persist_directory="./.chroma_db"
)

# retriever = Chroma(
#     collection_name="chroma_rag",
#     embedding_function=embeddings,
#     persist_directory="./.chroma_db"
# ).as_retriever()
#

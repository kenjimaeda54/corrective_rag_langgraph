import os

from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")


tavily = TavilySearch(
        tavily_api_key=tavily_api_key,
        max_results=3,
    )

def web_search(state: GraphState) -> Dict[str, Any]:

     question = state["question"]
     documents = state["documents"]

     tavily_web = tavily.invoke({"query": question})
     join_tavily_web = "\n".join([doc["content"] for doc in tavily_web["results"]])

     #tem que transformar em documents em nosso GraphState estamos
     #esperando uma lista de Documents
     web_results = Document(page_content=join_tavily_web)

     if web_results is not None:
         documents.append(web_results)
     else:
         documents = [web_results]
     return {"question": question, "documents": documents}


#para debugar
# if __name__ == "__main__":
#     web_search(GraphState({"question": "What is the capital of France?", "documents": []}))
#     print("oi")
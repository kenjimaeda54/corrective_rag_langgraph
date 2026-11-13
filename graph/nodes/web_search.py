from typing import Dict, Any

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

tavily = TavilySearch(
        max_results=3,
    )

def web_search(state: GraphState) -> Dict[str, Any]:

    question  = state["question"]
    docs = state["documents"]

    tavily_results = tavily.invoke({"query": question})
    tavily_results_text = "\n".join([doc["content"] for doc in tavily_results])
    web_results = Document(page_content=tavily_results_text)


    if web_results  is not None:
       docs.append(web_results)
    else:
       docs = [web_results]
    return {"question": question, "documents": docs}
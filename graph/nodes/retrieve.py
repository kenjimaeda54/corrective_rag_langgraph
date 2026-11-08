from typing import Dict, Any

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:

    question = state["question"]
    docs = retriever.invoke(question)
    return {"question": question, "documents": docs}


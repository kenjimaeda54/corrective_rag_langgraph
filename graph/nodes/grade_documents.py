from typing import Dict, Any


from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) ->  Dict[str, Any]:
    """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for document in documents:
        score = retrieval_grader.invoke({"document": document.page_content, "question": question})

        if score.binary_score.lower() == "yes":
            filtered_docs.append(document)
        else:
            web_search = True
            continue

    return {"question": question, "documents": filtered_docs, "web_search": web_search}




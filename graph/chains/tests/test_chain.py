from graph.chains.retriver_grader import GradeDocuments, retriever_grader
from ingestion import retriever


def retriever_document_yes():
     question = "agent memory"
     docs = retriever.invoke({"question": question})
     doc_txt = docs[1].page_content

     binary_code: GradeDocuments = retriever_grader.invoke(
         {"question": question, "document": doc_txt}
     )

     assert  binary_code.binary_score == "yes"


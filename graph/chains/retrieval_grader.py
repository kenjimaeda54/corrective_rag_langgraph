import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field, SecretStr

load_dotenv()

APIKEY_OPENROUTER = os.environ.get("OPEN_ROUTER_API_KEY","")


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )



llm = ChatOpenRouter(
     api_key=SecretStr(APIKEY_OPENROUTER),
     model="z-ai/glm-5.3-flash"
)


#Estamos usando o método llm.with_structured_output(GradeDocuments)
# ara garantir que a resposta gerada pelo modelo de linguagem siga
# exatamente o formato de dados que definimos em uma classe
# (geralmente usando Pydantic, como GradeDocuments).
structed_output =  llm.with_structured_output(GradeDocuments)


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

template_prompt = ChatPromptTemplate(
    [
        ("system",system),
        ("human","Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = template_prompt | structed_output


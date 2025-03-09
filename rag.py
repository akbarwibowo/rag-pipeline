from extract_and_save_file import vector_store

from dotenv import find_dotenv, load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

import logging
import os

try:
    load_dotenv(find_dotenv())
except:
    pass

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger()

logger.debug("Starting the RAG engine")

try:
    logger.info("Loading environment variables...")

    GROQ_APIKEY = os.environ.get("GROQ_APIKEY")
    chat_model = init_chat_model(
        "gemma2-9b-it", 
        model_provider="groq", 
        api_key=GROQ_APIKEY,
        temperature=0,
        )

except Exception as e:
    logger.error(f"Error: {e}")
    exit(1)



class ExpandOutputParser(BaseModel):
    """Use this format for creating expand query"""
    answers: list[str] = Field(description="List of expanded questions")



def expand_query(question):
    expand_parser = PydanticOutputParser(pydantic_object=ExpandOutputParser)
    query_expansion_prompt_template = """
    Translate the question into Indonesian, then generate 3 different version of the given question that would help in retrieving relevant information in Indonesian.
    The variations should include different terms and phrasings.

    original question: {question}

    {format_instruction}
    """

    query_expansion_prompt = PromptTemplate.from_template(template=query_expansion_prompt_template, partial_variables={"format_instruction": expand_parser.get_format_instructions()})
    query_expansion_chain = (
        {
            "question": RunnablePassthrough(),
        }
        | query_expansion_prompt
        | chat_model
        | expand_parser

    )

    query_lists = query_expansion_chain.invoke(question)
    
    return query_lists.answers


def get_response(question):
    expanded_queries = expand_query(question)

    all_documents = []

    for query in expanded_queries:
        documents = vector_store.max_marginal_relevance_search(
            query=query,
            k=3,
            fetch_k=20,
            lambda_mult=0.5
        )

        all_documents.extend(documents)

    prompt_template = """
    You are an expert question-answering assistant with access to specific documents. Your goal is to provide accurate, comprehensive answers based on the retrieved context.

    Retrieved context information is below.
    ---------------------
    {context}
    ---------------------

    Given the above context, please answer the following question in a detailed and structured way. If the answer cannot be found in the context, acknowledge this and provide the most helpful response possible without making up information.
    Understand the question and translate the question into Indonesian to match the context, then answer it straight away.
    Question: {question}

    Your answer should:
    1. Be directly relevant to the question
    2. Include specific facts from the context where applicable
    3. Cite sources when quoting information (mention PDF page numbers or CSV data when available only)
    4. Be well-structured and easy to understand
    5. Give answer with same language as the question

    Answer:
    """

    rag_prompt = PromptTemplate.from_template(template=prompt_template, partial_variables={"context": all_documents})

    rag_chain = (
        {
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | chat_model
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    return response


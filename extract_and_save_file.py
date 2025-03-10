from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

import logging
import os


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger()

# Log that the extraction process is starting
logger.debug("Starting the extraction process...")

try:
    logger.info("Loading environment variables...")

    # Load environment and object initialization
    load_dotenv(find_dotenv())
    GOOGLE_APIKEY = os.environ.get("GOOGLE_APIKEY")
    ASTRADB_APIKEY = os.environ.get("ASTRADB_APIKEY")
    ASTRADB_ENDPOINT = os.environ.get("ASTRADB_ENDPOINT")
    ASTRADB_COLLECTION_NAME = "first_vector_db"
    ASTRADB_NAMESPACE = "default_keyspace"

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_APIKEY)

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
    )

    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRADB_ENDPOINT,
        collection_name=ASTRADB_COLLECTION_NAME,
        token=ASTRADB_APIKEY,
        namespace=ASTRADB_NAMESPACE,
    )
except Exception as e:
    logger.error(f"Error: {e}")
    exit(1)


def load_pdf(file_path: str) -> list[Document]:
    """
    Loads a PDF file and extracts its contents.
    Args:
        file_path (str): The path to the PDF file to be loaded.
    Returns:
        list[Document]: A list of extracted PDF pages, where each page contains text and image content.
            Returns empty list if loading fails.
    Raises:
        Exception: If there is an error loading the PDF file. The error will be logged
                and the program will exit with status code 1.
    Note:
        This function uses PyPDFLoader with image extraction enabled and loads pages lazily
        to manage memory for large PDFs.
    """
    try:
        pdf_loader = PyPDFLoader(
            file_path=file_path,
            extract_images=True,
        )
        pdf_document = []

        logger.info("Loading pages...")

        for page in pdf_loader.lazy_load():
            pdf_document.append(page)
        
        logger.info(f"Total pages: {len(pdf_document)}")
        return pdf_document
        
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


def load_csv(file_path: str) -> list[Document]:
    """
    Load data from a CSV file and return it as a list of documents.
    Args:
        file_path (str): Path to the CSV file to be loaded
    Returns:
        list[Document]: List of documents where each document represents a row from the CSV
    Raises:
        Exception: If there is an error loading the CSV file, logs the error and exits
            with code 1
    """
    try:
        logger.info("Loading CSV...")


        csv_loader = CSVLoader(
            file_path=file_path,
        )

        csv_document = []
        for row in csv_loader.lazy_load():
            csv_document.append(row)
        
        logger.info(f"Total rows: {len(csv_document)}")
        return csv_document
    
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


def split_document(document: Document) -> list[Document]:
    """
    Split a PDF document into chunks using a text splitter.
    Args:
        document (Document): The PDF document to split.
            Can be either a single Document object or a list of Document objects.
    Returns:
        list[Document]: A list of Document chunks after splitting.
    Raises:
        Exception: If there is an error during the splitting process.
            Exits with code 1 if an error occurs.
    """
    try:
        logger.info("Splitting text...")
        
        splitted_document = text_splitter.split_documents(document)

        logger.info(f"Total chunks: {len(splitted_document)}")
        return splitted_document

    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


def save_to_vector_store(splitted_document: list[Document]) -> list[str]:
    """
    Saves documents into a vector store and returns their corresponding document IDs.
    This function takes a list of Document objects, stores them in a vector database,
    and returns the list of assigned document IDs.
    Args:
        splitted_document (list[Document]): A list of Document objects to be stored
            in the vector database.
    Returns:
        list[str]: A list of document IDs assigned by the vector store to the saved documents.
    Raises:
        Exception: If there is an error during the saving process, the function logs
            the error and exits the program.
    """
    try:
        logger.info("Saving to vector store...")
        document_ids = vector_store.add_documents(splitted_document)

        logger.info(f"Total documents saved: {len(document_ids)}")
        return document_ids
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


def extract_main(pdf_file_path: str, csv_file_path: str) -> list[str]:
    """
    Extracts text from a PDF file and CSV file, combines them, splits the combined data,
    and saves the resulting documents to a vector store.
    Args:
        pdf_file_path (str): Path to the PDF file to extract text from
        csv_file_path (str): Path to the CSV file to extract data from
    Returns:
        list[str]: List of document IDs generated when saving to vector store
    """
    document = load_pdf(pdf_file_path)
    data = load_csv(csv_file_path)

    combined_data = document + data
    splitted_document = split_document(combined_data)

    document_ids = save_to_vector_store(splitted_document)

    return document_ids

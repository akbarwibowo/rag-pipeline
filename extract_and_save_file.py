from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

import logging
import os


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

try:
    logger = logging.getLogger()
except:
    pass

logger.debug("Starting the extraction process...")

try:
    logger.info("Loading environment variables...")
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


def load_pdf(file_path):
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


def load_csv(file_path):
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


def split_pdf(document):
    try:
        logger.info("Splitting text...")
        
        splitted_document = text_splitter.split_documents(document)

        logger.info(f"Total chunks: {len(splitted_document)}")
        return splitted_document

    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


def save_to_vector_store(splitted_document):
    try:
        logger.info("Saving to vector store...")
        document_ids = vector_store.add_documents(splitted_document)

        logger.info(f"Total documents saved: {len(document_ids)}")
        return document_ids
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


def extract_main(pdf_file_path, csv_file_path):
    document = load_pdf(pdf_file_path)
    data = load_csv(csv_file_path)

    combined_data = document + data
    splitted_document = split_pdf(combined_data)

    document_ids = save_to_vector_store(splitted_document)

    return document_ids

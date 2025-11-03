from .ingest import ingest_documents
from .retrieve import retrieve_documents
from .generate import generate_responses


inngest_functions = [
    ingest_documents,
    retrieve_documents,
    generate_responses,
]

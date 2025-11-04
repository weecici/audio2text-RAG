from fastapi import APIRouter
from .ingest import router as ingest_router, ingest_documents
from .retrieve import router as retrieve_router, retrieve_documents
from .generate import router as generate_router, generate_responses

api_v1 = APIRouter()
api_v1.include_router(ingest_router)
api_v1.include_router(retrieve_router)
api_v1.include_router(generate_router)

inngest_functions = [
    ingest_documents,
    retrieve_documents,
    generate_responses,
]

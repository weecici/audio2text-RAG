import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="retrieve-documents",
    trigger=inngest.TriggerEvent(event="rag/retrieve-documents"),
    retries=0,
)
async def retrieve_documents(ctx: inngest.Context) -> dict[str, any]:
    return public_svcs.retrieve_documents(ctx).model_dump()


@router.post(
    "/retrieve",
    response_model=dict,
    summary="Retrieve relevant documents",
    description="Triggers an asynchronous job to retrieve documents based on the provided queries.",
)
async def retrieve(request: schemas.RetrievalRequest) -> dict[str, any]:
    result = await inngest_client.send(
        inngest.Event(name="rag/retrieve-documents", data=request.model_dump())
    )
    print(result)
    return {"message": "Retrieval process started"}

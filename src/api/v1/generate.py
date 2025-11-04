import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="generate-responses",
    trigger=inngest.TriggerEvent(event="rag/generate-responses"),
    retries=0,
)
async def generate_responses(ctx: inngest.Context) -> dict[str, any]:
    return public_svcs.generate_responses(ctx).model_dump()


@router.post(
    "/generate",
    response_model=dict,
    summary="Generate responses from documents",
    description="Triggers an asynchronous job to generate responses based on the provided queries and retrieved documents.",
)
async def generate(request: schemas.GenerationRequest) -> dict[str, any]:
    result = await inngest_client.send(
        inngest.Event(name="rag/generate-responses", data=request.model_dump())
    )
    print(result)
    return {"message": "Generation process started"}

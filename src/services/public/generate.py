import inngest
from fastapi import status
from src import schemas
from src.core import config
from src.services.internal import *
from .retrieve import retrieve_documents


def generate_responses(ctx: inngest.Context) -> schemas.GenerationResponse:
    try:
        request = schemas.GenerationRequest.model_validate(ctx.event.data)
        retrieved_docs = retrieve_documents(ctx).results

        return schemas.GenerationResponse(
            status=status.HTTP_200_OK,
            responses=[],
        )
    except Exception as e:
        ctx.logger.error(f"Error during generation process: {str(e)}")
        return schemas.GenerationResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            responses=[],
        )

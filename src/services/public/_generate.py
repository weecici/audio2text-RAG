from fastapi import status
from src import schemas
from src.services.internal import (
    get_augmented_prompts,
    generate,
    get_summarization_prompts,
    parse_summarization_responses,
)
from src.utils import logger
from ._retrieve import retrieve_documents


async def generate_responses(
    request: schemas.GenerationRequest,
) -> schemas.GenerationResponse:
    try:
        logger.info("Starting response generation process...")
        retrieved_docs = (await retrieve_documents(request)).results

        qa_prompts = get_augmented_prompts(
            queries=request.queries,
            contexts=retrieved_docs,
        )

        qa_responses = generate(
            prompts=qa_prompts,
            model=request.model_name,
        )

        sum_prompts = get_summarization_prompts(
            documents_list=retrieved_docs,
        )

        sum_responses = generate(
            prompts=sum_prompts,
            model=request.model_name,
        )

        parsed_summaries_list = parse_summarization_responses(
            responses=sum_responses,
            documents_list=retrieved_docs,
        )

        return schemas.GenerationResponse(
            status=status.HTTP_200_OK,
            responses=qa_responses,
            summarized_docs_list=parsed_summaries_list,
        )
    except Exception as e:
        logger.error(f"Error during generation process: {str(e)}")
        return schemas.GenerationResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            responses=[],
            summarized_docs_list=[],
        )

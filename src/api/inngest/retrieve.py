import inngest
from src import schemas
from src import services
from src.core import inngest_client, app


@inngest_client.create_function(
    fn_id="retrieve-documents",
    trigger=inngest.TriggerEvent(event="rag/retrieve-documents"),
    retries=0,
)
async def retrieve_documents(ctx: inngest.Context) -> schemas.RetrievalResponse:
    return services.retrieve_documents(ctx).model_dump()

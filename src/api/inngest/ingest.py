import inngest
from src import schemas
from src import services
from src.core import inngest_client, app


@inngest_client.create_function(
    fn_id="ingest-documents",
    trigger=inngest.TriggerEvent(event="rag/ingest-documents"),
    retries=0,
)
async def ingest_documents(ctx: inngest.Context) -> schemas.IngestionResponse:
    return services.ingest_documents(ctx).model_dump()

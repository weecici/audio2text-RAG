import inngest
import src.services.public as public_svcs
from src.core import inngest_client, app


@inngest_client.create_function(
    fn_id="generate-responses",
    trigger=inngest.TriggerEvent(event="rag/generate-responses"),
    retries=0,
)
async def generate_responses(ctx: inngest.Context) -> dict[str, any]:
    return public_svcs.generate_responses(ctx).model_dump()

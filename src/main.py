import inngest.fast_api
from src.core import inngest_client, app
from src.api.v1 import inngest_functions, api_v1


app.include_router(router=api_v1, prefix="/api/v1")
inngest.fast_api.serve(app=app, client=inngest_client, functions=inngest_functions)

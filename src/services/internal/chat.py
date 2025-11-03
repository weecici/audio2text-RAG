from functools import lru_cache
from cerebras.cloud.sdk import Cerebras
from src.core import config


@lru_cache(maxsize=1)
def _get_llm_client() -> Cerebras:
    print("Initializing Cerebras LLM client")
    client = Cerebras(api_key=config.CEREBRAS_API_KEY)
    return client


def generate(prompts: list[str], model: str) -> list[str]:
    client = _get_llm_client()
    return []

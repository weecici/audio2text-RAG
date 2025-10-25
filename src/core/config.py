import os
import random
from dotenv import load_dotenv, find_dotenv

load_dotenv()

rng = random.Random(42)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma-300m")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "google/embeddinggemma-300m")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

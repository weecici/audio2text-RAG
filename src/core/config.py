import os
import random
from dotenv import load_dotenv, find_dotenv

load_dotenv()

rng = random.Random(42)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma-300m")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "google/embeddinggemma-300m")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "bm25")
DISK_STORAGE_PATH = os.getenv("DISK_STORAGE_PATH", "./.storage")
WORD_PROCESS_METHOD = os.getenv("WORD_PROCESS_METHOD", "stem")
FUSION_METHOD = os.getenv("FUSION_METHOD", "dbsf")

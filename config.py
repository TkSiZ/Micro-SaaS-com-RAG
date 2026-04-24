import os

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "llama3"
TOP_K = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
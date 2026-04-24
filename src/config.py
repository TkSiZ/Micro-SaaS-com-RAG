import os

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "llama3"

CURRENT = os.path.abspath(__file__)
PARENT = os.path.dirname(CURRENT)
GRANDPARENT = os.path.dirname(PARENT)

VECTORSTORE_PATH = os.path.join(GRANDPARENT, "vectorstore")
COLLECTION_NAME = 'teoria_musical'
DATA_PATH = os.path.join(GRANDPARENT, "data", "raw")
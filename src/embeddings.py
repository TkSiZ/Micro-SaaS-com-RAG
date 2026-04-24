from sentence_transformers import SentenceTransformer
import chromadb

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))        # src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from config import EMBEDDING_MODEL, VECTORSTORE_PATH, COLLECTION_NAME

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
cliente_chroma = chromadb.PersistentClient(path=VECTORSTORE_PATH)
collection = cliente_chroma.get_or_create_collection(COLLECTION_NAME)

def index_chunks(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadata = [{"source" : c["source"]} for c in chunks]
    embeddings = embedding_model.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadata)
    print(f"{len(chunks)} chunks indexados")

def retrieval(question : str, top_k : int = 5) -> list[str]:
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings= question_embedding,
        n_results=top_k
        
    )

    return results["documents"][0]
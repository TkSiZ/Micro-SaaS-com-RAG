from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))        # src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from config import CHUNK_SIZE, CHUNK_OVERLAP

def create_chunks(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = []

    for doc in documents:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append({
                "id" : f"{doc['source']}_chunk_{i}",
                "text" : part,
                "source" : doc["source"]
            })
    
    return chunks
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))        # src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root


from ingestion import load_pdfs
from chunking import create_chunks
from embeddings import index_chunks
from config import DATA_PATH

if __name__ == "__main__":
    print("Carregando os pdfs")
    docs = load_pdfs(DATA_PATH)
    print("Criando os chunks")
    chunks = create_chunks(docs)
    print("Indexando e adicionando ao vectordb")
    index_chunks(chunks)
    print("Pipeline finalizada")
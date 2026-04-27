from src.pipeline import *

if __name__ == "__main__":
    from src.ingestion import load_pdfs
    from src.chunking import create_chunks
    from src.embeddings import index_chunks
    from src.config import DATA_PATH

    print("Carregando os PDFs...")
    docs = load_pdfs(DATA_PATH)
    print("Criando os chunks...")
    chunks = create_chunks(docs)
    print("Indexando no vectorstore...")
    index_chunks(chunks)
    print("Pipeline finalizada!")
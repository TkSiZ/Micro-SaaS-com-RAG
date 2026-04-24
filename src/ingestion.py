import fitz
import pdfplumber
import config as C
from pathlib import Path

def load_pdfs(folder: str) -> list[dict]:
    documents = []
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    pdf_files = list(folder_path.glob("*.pdf"))

    if not pdf_files:
        print("Nenhum PDF encontrado na pasta.")
        return []

    for pdf_path in pdf_files:
        print(f"Processando: {pdf_path.name}")
        text = ""

        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"PyMuPDF falhou em {pdf_path.name}: {e}")

        if not text.strip():
            print(f"Texto vazio com PyMuPDF, tentando pdfplumber...")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
            except Exception as e:
                print(f"pdfplumber falhou em {pdf_path.name}: {e}")

        if text.strip():
            print(f"OK — {len(text)} caracteres extraídos")
        else:
            print(f"AVISO — {pdf_path.name} ficou vazio (pode ser PDF escaneado)")

        documents.append({
            "source": pdf_path.name,
            "text": text.strip(),
            "path": str(pdf_path),
            "chars": len(text.strip())
        })

    print(f"\nTotal: {len(documents)} PDFs carregados")
    return documents

def main():
    docs = load_pdfs(C.DATA_PATH)

    for doc in docs:
        print(f"\nArquivo: {doc['source']}")
        print(f"Caracteres: {doc['chars']}")
        print(f"Prévia: {doc['text'][:200]}...")

if __name__ == "__main__":
    main()
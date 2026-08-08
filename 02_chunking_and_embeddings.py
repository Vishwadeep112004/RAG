import fitz
from sentence_transformers import SentenceTransformer

def get_chunks(text):
    chunk_size=1000
    chunks=[]
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def main():
    path = 'data/pdf_for_RAG.pdf'
    doc = fitz.open(path)
    text=""
    for page in doc:
        text+=page.get_text()
    chunks = get_chunks(text)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    print(type(embeddings))
    print(embeddings.shape)
    print(embeddings[:10])

if __name__=="__main__":
    main()

import fitz
from sentence_transformers import SentenceTransformer
import faiss

def get_chunks(text):
    n=1000
    chunks=[]
    for i in range(0,len(text),n):chunks.append(text[i:i+n])
    return chunks

def main():
    path='data/pdf_for_RAG.pdf'
    doc=fitz.open(path)
    text=""
    for page in doc:text+=page.get_text()
    chunks=get_chunks(text)
    model=SentenceTransformer('all-MiniLM-L6-v2')
    embeddings=model.encode(chunks,normalize_embeddings=True)
    dim=embeddings.shape[1]
    ind=faiss.IndexFlatIP(dim)
    ind.add(embeddings)
    print(ind.ntotal)
    query = "What is self attention?"
    query_embedding = model.encode([query],normalize_embeddings=True)
    k=5
    similarities,indices=ind.search(query_embedding,k)

    for i in range(k):
        index = indices[0][i]
        print("=" * 80)
        print(f"Rank       : {i + 1}")
        print(f"Chunk      : {index}")
        print(f"Similarity : {similarities[0][i]:.4f}")
        print("-" * 80)
        print(chunks[index])

if __name__=="__main__":main()
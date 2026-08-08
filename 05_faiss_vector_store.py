import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def get_chunks(text):
    n=1000
    chunks=[]
    for i in range(0,len(text),n): chunks.append(text[i:i+n])
    return chunks

def main():
    path='data/pdf_for_RAG.pdf'
    doc=fitz.open(path)
    text=""
    for page in doc:text+=page.get_text()
    chunks=get_chunks(text)
    model=SentenceTransformer('all-MiniLM-L6-v2')
    embeddings=model.encode(chunks)
    dim=embeddings.shape[1]
    ind=faiss.IndexFlatL2(dim)
    ind.add(embeddings)
    query="what is self attention?"
    query_embedding=model.encode([query])
    # print(query_embedding.shape)
    k=5
    distances,indices=ind.search(query_embedding,k)
    print(distances)
    print(indices)

    for i in range(k):
        index = indices[0][i]
        print("=" * 80)
        print(f"Rank : {i + 1}")
        print(f"Chunk Index : {index}")
        print(f"Distance : {distances[0][i]:.4f}")
        print("-" * 80)
        print(chunks[index])

if __name__=="__main__": main()

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

def get_chunks(text):
    chunks_size=1000
    chunks=[]
    for i in range(0, len(text), chunks_size):chunks.append(text[i:i+chunks_size])
    return chunks

def main():
    path='data/pdf_for_RAG.pdf'
    doc=fitz.open(path)
    text=""
    for page in doc:text+=page.get_text()
    chunks=get_chunks(text)
    model=SentenceTransformer("all-MiniLM-L6-v2")
    embeddings=model.encode(chunks)

    query="What is self attention?"
    query_embedding=model.encode(query)

    similarities=np.dot(query_embedding,embeddings.T)
    top_k = 5

    top_indices = np.argsort(similarities)[-top_k:]
    print(top_indices)
    top_indices = top_indices[::-1] 

    for index in top_indices:
        print(index, similarities[index])

if __name__=="__main__":
    main()
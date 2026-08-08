#here we only retireve the chunk which is best match 
# but what about that chunks that are close matches?
# therefore we take top k chunks and return them to the user

import fitz
from sentence_transformers import SentenceTransformer
import numpy as np

def get_chunks(text):
    chunk_size=1000
    chunks=[]
    for i in range(0, len(text), chunk_size): chunks.append(text[i:i+chunk_size])
    return chunks

def main():
    path='data/pdf_for_RAG.pdf'
    doc=fitz.open(path)
    text=""
    for page in doc: text+=page.get_text()
    chunks=get_chunks(text)
    model=SentenceTransformer("all-MiniLM-L6-v2")
    embeddings=model.encode(chunks)

    query="What is self attention?"
    query_embedding=model.encode(query)
    print(query_embedding.shape)
    similarities=np.dot(query_embedding, embeddings.T)

    best_match_index=np.argmax(similarities)
    print(f"Best Match Index: {best_match_index}")
    print(chunks[best_match_index])

if __name__=="__main__":
    main()

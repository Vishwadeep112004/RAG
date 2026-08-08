import fitz
from sentence_transformers import SentenceTransformer
import faiss
from google import genai

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
    x=1
    while x==1:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            x=0
            break   
        
        query_embedding = model.encode([query],normalize_embeddings=True)
        k=5
        similarities,indices=ind.search(query_embedding,k)
        context=""
        for i in range(k):
            index = indices[0][i]
            context+=chunks[index]
            context+="\n\n"
        prompt = f"""
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {query}
        """

        client = genai.Client(api_key="enter api key")

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        print(response.text)


if __name__=="__main__":main()
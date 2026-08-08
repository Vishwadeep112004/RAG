import fitz
pdf_path ='data/pdf_for_RAG.pdf'
doc = fitz.open(pdf_path)
# print(f"Total Pages: {len(doc)}")
page_one=doc.load_page(0)
text=page_one.get_text()
# print(text)

def get_chunks(text):
    chunk_size = 1000
    chunks = []
    for i in range(0, len(text),chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks
#we can print document by
fulltext = ""
for page in doc:
    fulltext += page.get_text()
    # print(page.get_text())

# print("=" * 50)
# print(f"Characters : {len(fulltext)}")
# print(f"Words      : {len(fulltext.split())}")

chunks = get_chunks(fulltext)
print(f"Total Chunks: {len(chunks)}")
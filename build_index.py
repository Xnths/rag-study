from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def is_valid_chunk(doc):
    text = doc.page_content
    if len(text.strip()) < 100:
        return False
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\t')
    ratio = control_chars / len(text)
    if ratio > 0.02:
        return False
    return True

loader = PyPDFLoader("./arquivo.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(pages)

chunks_before = len(chunks)
chunks = [c for c in chunks if is_valid_chunk(c)]
chunks_after = len(chunks)

print(f"Chunks before filter: {chunks_before}")
print(f"Chunks after filter:  {chunks_after}")
print(f"Removed:              {chunks_before - chunks_after}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"Chunks indexed: {vectorstore.index.ntotal}")
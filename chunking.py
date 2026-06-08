# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./arquivo.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(pages)

print(f"Pages:  {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"\nChunk 0 metadata: {chunks[0].metadata}")
print(f"Chunk 0 length:   {len(chunks[0].page_content)}")
print(f"Chunk 0 content:\n{chunks[0].page_content}")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./arquivo.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

for i, c in enumerate(chunks):
    if c.metadata["page"] == 1:
        print(f"--- chunk {i} ---")
        print(repr(c.page_content[:200]))
        print()
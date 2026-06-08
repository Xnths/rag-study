# pyrefly: ignore [missing-import]
from langchain_ollama import OllamaEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

queries = [
    "What is the difference between RAG-Sequence and RAG-Token?",
    "How does DPR retrieve documents?",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = retriever.invoke(q)
    for r in results:
        print(f"  [page {r.metadata['page']}] {r.page_content[:150]}")
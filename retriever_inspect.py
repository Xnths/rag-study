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
    "How does the encoder work?",
    "What is FAISS used for?",
    "What is the capital of Brazil?",
    "How does fine-tuning work?",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = retriever.invoke(q)
    for r in results:
        print(f"  [{r.metadata['index']}] {r.page_content}")
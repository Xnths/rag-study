# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_ollama import OllamaEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS

sentences = [
    "The transformer architecture uses an encoder and a decoder.",
    "The encoder maps an input sequence to a vector in R^d.",
    "The decoder generates tokens autoregressively from that vector.",
    "BERT is an encoder-only model pretrained with masked language modeling.",
    "BART is an encoder-decoder model pretrained with a denoising objective.",
    "Dense Passage Retrieval uses two BERT encoders: one for queries, one for documents.",
    "FAISS enables approximate maximum inner product search in sublinear time.",
    "The RAG model marginalizes over retrieved documents to generate answers.",
    "Fine-tuning updates model parameters on a supervised dataset.",
    "The document index stores precomputed embeddings for all documents in the corpus.",
]

docs = [
    Document(page_content=s, metadata={"source": "corpus", "index": i})
    for i, s in enumerate(sentences)
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

print(type(vectorstore))
print(vectorstore.index.ntotal)

query = "How does the encoder work?"
results = vectorstore.similarity_search(query, k=3)

for r in results:
    print(r.metadata["index"], r.page_content)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

results = retriever.invoke("How does the encoder work?")
print(type(results))
print(len(results))
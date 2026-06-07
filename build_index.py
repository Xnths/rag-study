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
    "FAISS is used for fast similarity search over large collections of vectors.",
    "FAISS finds the most similar vectors to a query vector using inner product.",
    "The RAG model marginalizes over retrieved documents to generate answers.",
    "Fine-tuning updates model parameters on a supervised dataset.",
    "The document index stores precomputed embeddings for all documents in the corpus.",
    "RAG retrieves documents from a corpus and uses them as context for generation.",
    "The retriever scores documents by computing the inner product between query and document embeddings.",
    "Top-K retrieval selects the K documents with the highest similarity score.",
]

docs = [
    Document(page_content=s, metadata={"source": "corpus", "index": i})
    for i, s in enumerate(sentences)
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
print("Index saved.")
print(f"Total vectors: {vectorstore.index.ntotal}")
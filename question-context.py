# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_ollama import OllamaEmbeddings, ChatOllama
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnableParallel, RunnableLambda
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    "If the answer is not in the context, say 'I don't know'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

llm = ChatOllama(model="llama3.2")
parser = StrOutputParser()

chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x),
    })
    | prompt
    | llm
    | parser
)

answer = chain.invoke("What is FAISS used for?")
print(answer)

answer = chain.invoke("What is the capital of Brazil?")
print(answer)

answer = chain.invoke("What is my name?")
print(answer)

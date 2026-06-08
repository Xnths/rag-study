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

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

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

queries = [
    "What is the difference between RAG-Sequence and RAG-Token?",
    "How does DPR retrieve documents?",
]

for q in queries:
    print(f"\nQuestion: {q}")
    print(chain.invoke(q))
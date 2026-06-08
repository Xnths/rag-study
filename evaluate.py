import re
import string
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
eval_pairs = [
    {
        "question": "What does RAG stand for?",
        "answer": "Retrieval-Augmented Generation"
    },
    {
        "question": "What is the non-parametric memory in RAG?",
        "answer": "a dense vector index of Wikipedia"
    },
    {
        "question": "What model is used as the generator in RAG?",
        "answer": "BART"
    },
    {
        "question": "What is MIPS?",
        "answer": "Maximum Inner Product Search"
    },
    {
        "question": "What dataset has 21 million documents in RAG?",
        "answer": "Wikipedia"
    },
]

def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)

def exact_match(y_pred, y_ref):
    return int(normalize(y_pred) == normalize(y_ref))

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    "Answer in as few words as possible. Do not use complete sentences. "
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

scores = []
for pair in eval_pairs:
    q = pair["question"]
    ref = pair["answer"]

    docs = retriever.invoke(q)
    context = format_docs(docs)
    answer_in_context = normalize(ref) in normalize(context)

    y_pred = chain.invoke(q)
    em = exact_match(y_pred, ref)
    scores.append(em)

    print(f"Q: {q}")
    print(f"Expected:         {ref}")
    print(f"Predicted:        {y_pred}")
    print(f"Answer in z:      {answer_in_context}")
    print(f"EM:               {em}")
    print()

print(f"Average EM: {sum(scores)/len(scores):.2f}")
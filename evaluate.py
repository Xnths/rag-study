import string
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
# pyrefly: ignore [missing-import]
from eval_dataset import eval_pairs

def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)

def exact_match(y_pred, y_ref):
    return int(normalize(y_pred) == normalize(y_ref))

def lcs_length(a, b):
    a = a.split()
    b = b.split()
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(y_pred, y_ref):
    y_pred = normalize(y_pred)
    y_ref = normalize(y_ref)
    pred_tokens = y_pred.split()
    ref_tokens = y_ref.split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    lcs = lcs_length(y_pred, y_ref)
    P = lcs / len(pred_tokens)
    R = lcs / len(ref_tokens)
    if P + R == 0:
        return 0.0
    return (2 * P * R) / (P + R)

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

em_scores = []
rl_scores = []

for pair in eval_pairs:
    q = pair["question"]
    ref = pair["answer"]
    y_pred = chain.invoke(q)
    em = exact_match(y_pred, ref)
    rl = rouge_l(y_pred, ref)
    em_scores.append(em)
    rl_scores.append(rl)
    print(f"Q:         {q}")
    print(f"Expected:  {ref}")
    print(f"Predicted: {y_pred}")
    print(f"EM:        {em}  |  Rouge-L: {rl:.2f}")
    print()

print(f"Average EM:      {sum(em_scores)/len(em_scores):.2f}")
print(f"Average Rouge-L: {sum(rl_scores)/len(rl_scores):.2f}")
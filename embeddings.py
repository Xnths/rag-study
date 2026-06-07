# pyrefly: ignore [missing-import]
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = embeddings.embed_query("What is a token?")

print(type(vector))
print(len(vector))
print(vector[:5])

v1 = embeddings.embed_query("encoder maps input to a vector")
v2 = embeddings.embed_query("decoder generates tokens autoregressively")
v3 = embeddings.embed_query("encoder produces a dense representation")

dot_12 = sum(a * b for a, b in zip(v1, v2))
dot_13 = sum(a * b for a, b in zip(v1, v3))

print(f"dot(v1, v2) = {dot_12:.4f}")
print(f"dot(v1, v3) = {dot_13:.4f}")
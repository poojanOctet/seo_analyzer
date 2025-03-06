from sentence_transformers import SentenceTransformer

EMBEDDER = SentenceTransformer("all-mpnet-base-v2")
input = "Your text string goes here"

embeddings = EMBEDDER.encode(input)

print(f"Embeddings: {embeddings}")
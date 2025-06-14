# ......... IMPORTS ..........
import numpy as np
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import re
import os
import pyarrow.parquet as pq


def load_text(path):
    document = Document(path)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def chunk_text(text):
    return re.split(r"(?<=[.؟!?،؛:])\s+", text)


DOCX_PATH = "input.docx"
PARQUET_PATH = "output_embeddings.parquet"
MODEL = "all-MiniLM-L6-v2"
TOP_PRINTS = 5
FORCE_REGEN_PARQUET = True

model = SentenceTransformer(MODEL)

if FORCE_REGEN_PARQUET or os.path.exists(PARQUET_PATH):
    print("Generating new embeddings...")
    raw_text = load_text(DOCX_PATH)
    texts = chunk_text(raw_text)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    df = pd.DataFrame({"text": texts, "embeddings": embeddings})
    df.to_parquet(PARQUET_PATH, index=False)
else:
    print("Loading existing parquet file...")
    df = pd.read_parquet(PARQUET_PATH)

embedding_matrix = np.vstack(df["embeddings"].values).astype("float32")
DIM = embedding_matrix.shape[1]

index = AnnoyIndex(DIM, 'dot')
for i, vec in enumerate(embedding_matrix):
    index.add_item(i, vec)
index.build(10)


def semantic_search(query, top_k=TOP_PRINTS):
    q_vec = model.encode([query])[0].astype('float32')
    annoy_idx = index.get_nns_by_vector(q_vec, 1)[0]
    scores = embedding_matrix @ q_vec

    best_index = int(np.argmax(scores))
    print(f"\nTop {top_k} dot products:")
    for rank, i in enumerate(np.argsort(scores)[::-1][:top_k], 1):
        flag = " <-- Annoy Pick: " if i == annoy_idx else ""
        if i == 1:
            print("MAX DOT PRODUCT: ")

        print(f"{rank:>2}. score = {scores[i]:.4f} | Chunk #{i}: {df.loc[i, 'text'][:80]} {flag}")

    return df.loc[annoy_idx, "text"]


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() == 'exit':
            break
        answer = semantic_search(query)
        print(f"\nAnswer: \n{answer}")

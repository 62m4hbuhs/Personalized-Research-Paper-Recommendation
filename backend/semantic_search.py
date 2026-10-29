import json
import faiss
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "allenai/scibert_scivocab_uncased"

INDEX_FILE = "data/processed/faiss/scibert.index"
PAPER_IDS_FILE = "data/processed/faiss/paper_id_mapping.json"
PAPERS_FILE = "data/processed/papers_clean.csv"

TOP_K = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(INDEX_FILE)

print(f"FAISS vectors: {index.ntotal:,}")
print(f"Vector dimension: {index.d}")


# ============================================================
# LOAD PAPER IDs
# ============================================================

with open(
    PAPER_IDS_FILE,
    "r",
    encoding="utf-8"
) as f:
    paper_ids = json.load(f)


# ============================================================
# LOAD PAPER METADATA
# ============================================================

print("Loading paper metadata...")

papers = pd.read_csv(
    PAPERS_FILE,
    usecols=[
        "paperId",
        "title",
        "abstract",
        "year",
        "citationCount"
    ]
)

papers = papers.drop_duplicates(
    subset="paperId"
)

papers_lookup = papers.set_index(
    "paperId"
).to_dict("index")


# ============================================================
# LOAD SCIBERT
# ============================================================

print("\nLoading SciBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("SciBERT ready.")


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    last_hidden_state,
    attention_mask
):

    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ============================================================
# QUERY EMBEDDING
# ============================================================

def create_query_embedding(query):

    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        outputs = model(**encoded)

        embedding = mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

    embedding = (
        embedding
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Same normalization used when building FAISS
    faiss.normalize_L2(embedding)

    return embedding


# ============================================================
# SEARCH
# ============================================================

def semantic_search(query, top_k=10):

    query_embedding = create_query_embedding(
        query
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        paper_id = paper_ids[idx]

        metadata = papers_lookup.get(
            paper_id
        )

        if metadata is None:
            continue

        results.append({
            "paperId": paper_id,
            "score": float(score),
            "title": metadata["title"],
            "year": metadata["year"],
            "citationCount": metadata["citationCount"]
        })

    return results


# ============================================================
# INTERACTIVE SEARCH
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SCI-BERT SEMANTIC SEARCH")
    print("=" * 70)

    query = input(
        "\nEnter your research topic/query:\n> "
    )

    results = semantic_search(
        query,
        TOP_K
    )

    print("\n" + "=" * 70)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{rank}. {result['title']}"
        )

        print(
            f"   Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"   Year: "
            f"{result['year']}"
        )

        print(
            f"   Citations: "
            f"{result['citationCount']}"
        )

        print(
            f"   Paper ID: "
            f"{result['paperId']}"
        )
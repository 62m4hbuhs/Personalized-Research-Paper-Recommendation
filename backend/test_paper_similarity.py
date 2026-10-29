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

TARGET_TITLE = "How Powerful are Graph Neural Networks?"


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
# LOAD DATA
# ============================================================

print("\nLoading paper metadata...")

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

papers["title_clean"] = (
    papers["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

matches = papers[
    papers["title_clean"].str.lower()
    == TARGET_TITLE.lower()
]

if len(matches) == 0:
    print("Target paper not found.")
    raise SystemExit

target = matches.iloc[0]

print("\nTarget paper:")
print(target["title"])
print(f"Paper ID: {target['paperId']}")


if pd.isna(target["abstract"]) or not str(
    target["abstract"]
).strip():

    print("Target paper has no abstract.")
    raise SystemExit


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)

with open(
    PAPER_IDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    paper_ids = json.load(f)


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
# CREATE EMBEDDING
# ============================================================

text = str(target["abstract"])

encoded = tokenizer(
    [text],
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

faiss.normalize_L2(embedding)


# ============================================================
# SEARCH
# ============================================================

scores, indices = index.search(
    embedding,
    TOP_K + 1
)


# ============================================================
# RESULTS
# ============================================================

lookup = (
    papers
    .drop_duplicates("paperId")
    .set_index("paperId")
)

print("\n" + "=" * 70)
print("PAPER-TO-PAPER SEMANTIC SIMILARITY")
print("=" * 70)

rank = 1

for score, idx in zip(
    scores[0],
    indices[0]
):

    paper_id = paper_ids[idx]

    # Don't recommend the source paper itself
    if paper_id == target["paperId"]:
        continue

    if paper_id not in lookup.index:
        continue

    result = lookup.loc[paper_id]

    print(f"\n{rank}. {result['title']}")
    print(f"   Similarity: {score:.4f}")
    print(f"   Year: {result['year']}")
    print(f"   Citations: {result['citationCount']}")
    print(f"   Paper ID: {paper_id}")

    rank += 1

    if rank > TOP_K:
        break
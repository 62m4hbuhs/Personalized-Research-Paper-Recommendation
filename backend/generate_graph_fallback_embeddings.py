import os
import json
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

GRAPH_NODES_FILE = "data/processed/graph/node_mapping.csv"
PAPERS_FILE = "data/processed/papers_clean.csv"

EXISTING_IDS_FILE = (
    "data/processed/embeddings/paper_ids.json"
)

OUTPUT_EMBEDDINGS = (
    "data/processed/graph/fallback_embeddings.npy"
)

OUTPUT_IDS = (
    "data/processed/graph/fallback_paper_ids.json"
)

MODEL_NAME = "allenai/scibert_scivocab_uncased"

BATCH_SIZE = 8
MAX_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("GRAPH FALLBACK EMBEDDING GENERATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(f"Batch size: {BATCH_SIZE}")
print(f"Max length: {MAX_LENGTH}")


# ============================================================
# LOAD GRAPH NODES
# ============================================================

print("\nLoading graph nodes...")

graph_nodes = pd.read_csv(
    GRAPH_NODES_FILE
)

graph_nodes = graph_nodes.sort_values(
    "node_id"
).reset_index(drop=True)

print(
    f"Graph nodes: {len(graph_nodes):,}"
)


# ============================================================
# LOAD EXISTING EMBEDDING IDs
# ============================================================

print("\nLoading existing embedding IDs...")

with open(
    EXISTING_IDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    existing_ids = set(
        json.load(f)
    )

print(
    f"Existing SciBERT embeddings: "
    f"{len(existing_ids):,}"
)


# ============================================================
# LOAD PAPER TITLES
# ============================================================

print("\nLoading paper titles...")

papers = pd.read_csv(
    PAPERS_FILE,
    usecols=[
        "paperId",
        "title"
    ]
)

papers = papers.drop_duplicates(
    subset="paperId"
)

papers["title"] = (
    papers["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

paper_lookup = papers.set_index(
    "paperId"
)["title"].to_dict()


# ============================================================
# FIND MISSING GRAPH NODES
# ============================================================

missing = graph_nodes[
    ~graph_nodes["paperId"].isin(
        existing_ids
    )
].copy()

print("\n" + "=" * 70)
print("FALLBACK TARGETS")
print("=" * 70)

print(
    f"Graph nodes without existing embedding: "
    f"{len(missing):,}"
)


# ============================================================
# KEEP ONLY NODES WITH TITLES
# ============================================================

missing["title"] = missing[
    "paperId"
].map(paper_lookup)

missing["title"] = (
    missing["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missing = missing[
    missing["title"] != ""
].copy()

print(
    f"Missing nodes with usable titles: "
    f"{len(missing):,}"
)

print(
    f"Missing nodes without usable titles: "
    f"{len(graph_nodes) - len(existing_ids) - len(missing):,}"
)


# ============================================================
# LOAD SCIBERT
# ============================================================

print("\nLoading SciBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading SciBERT model...")

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

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

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
# GENERATE FALLBACK EMBEDDINGS
# ============================================================

fallback_ids = []
fallback_embeddings = []

total = len(missing)

print("\nGenerating title embeddings...")
print("=" * 70)

with torch.inference_mode():

    for start in tqdm(
        range(
            0,
            total,
            BATCH_SIZE
        ),
        desc="Title embeddings"
    ):

        batch = missing.iloc[
            start:start + BATCH_SIZE
        ]

        titles = batch[
            "title"
        ].tolist()

        encoded = tokenizer(
            titles,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(
            **encoded
        )

        embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        fallback_embeddings.append(
            embeddings
        )

        fallback_ids.extend(
            batch["paperId"].tolist()
        )

        del encoded
        del outputs
        del embeddings

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# COMBINE
# ============================================================

print("\nCombining fallback embeddings...")

if fallback_embeddings:

    fallback_matrix = np.vstack(
        fallback_embeddings
    )

else:

    fallback_matrix = np.empty(
        (0, 768),
        dtype=np.float32
    )


print(
    f"Fallback embedding shape: "
    f"{fallback_matrix.shape}"
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving fallback embeddings...")

os.makedirs(
    os.path.dirname(
        OUTPUT_EMBEDDINGS
    ),
    exist_ok=True
)

np.save(
    OUTPUT_EMBEDDINGS,
    fallback_matrix
)

with open(
    OUTPUT_IDS,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        fallback_ids,
        f
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FALLBACK EMBEDDINGS COMPLETE")
print("=" * 70)

print(
    f"Fallback embeddings: "
    f"{len(fallback_ids):,}"
)

print(
    f"Dimension: "
    f"{fallback_matrix.shape[1] if len(fallback_matrix) else 768}"
)

print("\nSaved:")

print(
    f" - {OUTPUT_EMBEDDINGS}"
)

print(
    f" - {OUTPUT_IDS}"
)

print("\nDone.")
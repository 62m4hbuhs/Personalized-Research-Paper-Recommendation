import os
import json
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# CONFIGURATION

INPUT_FILE = "data/processed/papers_clean.csv"
OUTPUT_DIR = "data/processed/embeddings"

MODEL_NAME = "allenai/scibert_scivocab_uncased"

BATCH_SIZE = 4

# SciBERT maximum sequence length
MAX_LENGTH = 256

# Save checkpoint every N batches
CHECKPOINT_EVERY = 100



# DEVICE


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("SciBERT Embedding Generation")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max length: {MAX_LENGTH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)



# LOAD DATA


print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["paperId", "title", "abstract"]
)

print(f"Total papers: {len(df):,}")

# Remove papers without abstracts
df["abstract"] = df["abstract"].fillna("").astype(str)

usable_df = df[df["abstract"].str.strip() != ""].copy()

print(f"Papers with abstracts: {len(usable_df):,}")
print(f"Papers without abstracts: {len(df) - len(usable_df):,}")


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

print("SciBERT loaded successfully.")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings while ignoring padding tokens.
    """

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
# GENERATE EMBEDDINGS
# ============================================================

paper_ids = []
embeddings = []

total = len(usable_df)

print("\nStarting embedding generation...")
print("=" * 70)

start_time = time.time()

with torch.inference_mode():

    for start in tqdm(
        range(0, total, BATCH_SIZE),
        desc="Generating embeddings"
    ):

        batch = usable_df.iloc[
            start:start + BATCH_SIZE
        ]

        texts = batch["abstract"].tolist()

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        batch_embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

        # Move embeddings back to CPU
        batch_embeddings = (
            batch_embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        embeddings.append(batch_embeddings)

        paper_ids.extend(
            batch["paperId"].tolist()
        )

        # Free GPU tensors
        del encoded
        del outputs
        del batch_embeddings

        # Periodic GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# COMBINE
# ============================================================

print("\nCombining embeddings...")

embedding_matrix = np.vstack(embeddings)

print(f"Embedding shape: {embedding_matrix.shape}")
print(f"Embedding dtype: {embedding_matrix.dtype}")


# ============================================================
# SAVE
# ============================================================

print("\nSaving embeddings...")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "scibert_embeddings.npy"
    ),
    embedding_matrix
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "paper_ids.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        paper_ids,
        f
    )


# Save metadata
metadata = {
    "model": MODEL_NAME,
    "device": str(DEVICE),
    "embedding_dimension": int(
        embedding_matrix.shape[1]
    ),
    "number_of_embeddings": int(
        embedding_matrix.shape[0]
    ),
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "pooling": "mean_pooling",
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 70)

print(f"Total papers in dataset:     {len(df):,}")
print(f"Papers with abstracts:       {len(usable_df):,}")
print(f"Embeddings generated:        {len(embedding_matrix):,}")
print(f"Embedding dimension:         {embedding_matrix.shape[1]}")
print(f"Embedding dtype:             {embedding_matrix.dtype}")
print(f"Time taken:                  {elapsed / 60:.2f} minutes")

print("\nSaved files:")

print(
    " - data/processed/embeddings/scibert_embeddings.npy"
)

print(
    " - data/processed/embeddings/paper_ids.json"
)

print(
    " - data/processed/embeddings/metadata.json"
)

print("\nDone.")
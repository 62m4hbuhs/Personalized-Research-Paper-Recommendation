import os
import json
import numpy as np
import faiss


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS_FILE = "data/processed/embeddings/scibert_embeddings.npy"
PAPER_IDS_FILE = "data/processed/embeddings/paper_ids.json"

OUTPUT_DIR = "data/processed/faiss"

INDEX_FILE = os.path.join(
    OUTPUT_DIR,
    "scibert.index"
)

INDEX_METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 70)
print("FAISS Index Construction")
print("=" * 70)

print("\nLoading SciBERT embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
)

print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dtype: {embeddings.dtype}")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

print("\nNormalizing embeddings...")

embeddings = embeddings.astype(
    np.float32,
    copy=False
)

faiss.normalize_L2(embeddings)

print("Normalization complete.")


# ============================================================
# CREATE INDEX
# ============================================================

dimension = embeddings.shape[1]

print(f"\nEmbedding dimension: {dimension}")

# Inner Product + normalized vectors = cosine similarity
index = faiss.IndexFlatIP(dimension)

print("Adding embeddings to FAISS...")

index.add(embeddings)

print(f"FAISS index size: {index.ntotal:,}")


# ============================================================
# SAVE INDEX
# ============================================================

print("\nSaving FAISS index...")

faiss.write_index(
    index,
    INDEX_FILE
)


# ============================================================
# SAVE PAPER ID MAPPING
# ============================================================

print("Loading paper IDs...")

with open(
    PAPER_IDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    paper_ids = json.load(f)


if len(paper_ids) != index.ntotal:
    raise ValueError(
        f"Mismatch: {len(paper_ids)} paper IDs "
        f"but {index.ntotal} embeddings."
    )


# FAISS row number → Semantic Scholar paper ID
mapping_file = os.path.join(
    OUTPUT_DIR,
    "paper_id_mapping.json"
)

with open(
    mapping_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        paper_ids,
        f
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "index_type": "IndexFlatIP",
    "similarity": "cosine",
    "dimension": dimension,
    "number_of_vectors": index.ntotal,
    "embedding_model": "allenai/scibert_scivocab_uncased",
    "normalized": True
}

with open(
    INDEX_METADATA_FILE,
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

print("\n" + "=" * 70)
print("FAISS INDEX COMPLETE")
print("=" * 70)

print(f"Vectors:        {index.ntotal:,}")
print(f"Dimensions:     {dimension}")
print("Similarity:     Cosine similarity")
print("Index type:     IndexFlatIP")

print("\nSaved:")

print(f" - {INDEX_FILE}")
print(f" - {mapping_file}")
print(f" - {INDEX_METADATA_FILE}")

print("\nDone.")
import os
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

GRAPH_NODES_FILE = "data/processed/graph/node_mapping.csv"

EMBEDDINGS_FILE = (
    "data/processed/embeddings/scibert_embeddings.npy"
)

EMBEDDING_IDS_FILE = (
    "data/processed/embeddings/paper_ids.json"
)

OUTPUT_DIR = "data/processed/graph"

OUTPUT_FEATURES = os.path.join(
    OUTPUT_DIR,
    "node_features.npy"
)

OUTPUT_MAPPING = os.path.join(
    OUTPUT_DIR,
    "feature_mapping.csv"
)


# ============================================================
# LOAD GRAPH NODE MAPPING
# ============================================================

print("=" * 70)
print("GRAPH FEATURE ALIGNMENT")
print("=" * 70)

print("\nLoading graph node mapping...")

nodes = pd.read_csv(
    GRAPH_NODES_FILE
)

print(f"Graph nodes: {len(nodes):,}")

print("\nColumns:")
print(list(nodes.columns))


# ============================================================
# VALIDATE NODE MAPPING
# ============================================================

required_columns = {
    "node_id",
    "paperId"
}

missing = required_columns - set(nodes.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if nodes["node_id"].duplicated().any():
    raise ValueError(
        "Duplicate node_id values found."
    )

if nodes["paperId"].duplicated().any():
    raise ValueError(
        "Duplicate paperId values found."
    )


nodes = nodes.sort_values(
    "node_id"
).reset_index(drop=True)


# ============================================================
# LOAD SCIBERT EMBEDDINGS
# ============================================================

print("\nLoading SciBERT embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE,
    mmap_mode="r"
)

print(
    f"SciBERT embedding shape: "
    f"{embeddings.shape}"
)


# ============================================================
# LOAD EMBEDDING PAPER IDs
# ============================================================

print("\nLoading embedding paper IDs...")

with open(
    EMBEDDING_IDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    embedding_paper_ids = json.load(f)

print(
    f"Embedding paper IDs: "
    f"{len(embedding_paper_ids):,}"
)


# ============================================================
# VALIDATE EMBEDDING DATA
# ============================================================

if len(embedding_paper_ids) != len(embeddings):

    raise ValueError(
        "Number of paper IDs does not match "
        "number of embeddings."
    )

if len(set(embedding_paper_ids)) != len(
    embedding_paper_ids
):

    raise ValueError(
        "Duplicate paper IDs found in embedding mapping."
    )


# ============================================================
# CREATE EMBEDDING LOOKUP
# ============================================================

print("\nCreating embedding lookup...")

embedding_lookup = {
    paper_id: idx
    for idx, paper_id in enumerate(
        embedding_paper_ids
    )
}


# ============================================================
# ALIGN GRAPH NODES WITH EMBEDDINGS
# ============================================================

print("\nAligning graph nodes...")

nodes["embedding_row"] = nodes[
    "paperId"
].map(embedding_lookup)


# ============================================================
# COVERAGE
# ============================================================

has_embedding = nodes[
    "embedding_row"
].notna()

with_embedding = int(
    has_embedding.sum()
)

without_embedding = int(
    (~has_embedding).sum()
)

total_nodes = len(nodes)

print("\n" + "=" * 70)
print("FEATURE COVERAGE")
print("=" * 70)

print(
    f"Total graph nodes:       {total_nodes:,}"
)

print(
    f"With SciBERT features:   {with_embedding:,}"
)

print(
    f"Without SciBERT features:{without_embedding:,}"
)

print(
    f"Coverage:                "
    f"{with_embedding / total_nodes * 100:.2f}%"
)


# ============================================================
# CHECK EXPECTED COVERAGE
# ============================================================

expected = 56433

if with_embedding != expected:

    print(
        f"\nWARNING: Expected approximately "
        f"{expected:,} embeddings but found "
        f"{with_embedding:,}."
    )


# ============================================================
# BUILD FULL NODE FEATURE MATRIX
# ============================================================

dimension = embeddings.shape[1]

print(
    f"\nCreating feature matrix: "
    f"({total_nodes:,}, {dimension})"
)

# Start with zeros for nodes without abstracts.
node_features = np.zeros(
    (total_nodes, dimension),
    dtype=np.float32
)


# Copy available SciBERT embeddings
available_rows = nodes[
    has_embedding
]["embedding_row"].astype(int).to_numpy()

available_node_ids = nodes[
    has_embedding
]["node_id"].astype(int).to_numpy()

print(
    f"Copying {len(available_rows):,} "
    f"SciBERT embeddings..."
)

node_features[
    available_node_ids
] = embeddings[
    available_rows
]


# ============================================================
# VERIFY
# ============================================================

print("\nRunning validation...")

if node_features.shape != (
    total_nodes,
    dimension
):

    raise ValueError(
        "Feature matrix has incorrect shape."
    )

if not np.isfinite(
    node_features
).all():

    raise ValueError(
        "Feature matrix contains NaN or Inf."
    )


# Verify known embedding rows
sample_count = min(
    100,
    len(available_rows)
)

sample_embedding_rows = available_rows[
    :sample_count
]

sample_node_ids = available_node_ids[
    :sample_count
]

if not np.allclose(
    node_features[sample_node_ids],
    embeddings[sample_embedding_rows]
):

    raise ValueError(
        "Embedding alignment validation failed."
    )

print(
    "Sample alignment check: PASSED"
)


# ============================================================
# SAVE FEATURE MATRIX
# ============================================================

print("\nSaving node features...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.save(
    OUTPUT_FEATURES,
    node_features
)


# ============================================================
# SAVE MAPPING
# ============================================================

mapping = nodes[
    [
        "node_id",
        "paperId",
        "embedding_row"
    ]
].copy()

mapping["has_embedding"] = (
    mapping["embedding_row"].notna()
)

mapping.to_csv(
    OUTPUT_MAPPING,
    index=False
)


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics = {
    "total_graph_nodes": total_nodes,
    "nodes_with_scibert": with_embedding,
    "nodes_without_scibert": without_embedding,
    "scibert_coverage_percent":
        round(
            with_embedding /
            total_nodes * 100,
            4
        ),
    "feature_dimension": dimension,
    "dtype": "float32",
    "missing_feature_strategy":
        "zero_vector"
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "feature_statistics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ALIGNMENT COMPLETE")
print("=" * 70)

print(
    f"Nodes:                 {total_nodes:,}"
)

print(
    f"Feature dimension:     {dimension}"
)

print(
    f"SciBERT features:      {with_embedding:,}"
)

print(
    f"Missing features:      {without_embedding:,}"
)

print(
    f"Coverage:              "
    f"{with_embedding / total_nodes * 100:.2f}%"
)

print("\nSaved:")

print(
    f" - {OUTPUT_FEATURES}"
)

print(
    f" - {OUTPUT_MAPPING}"
)

print(
    " - data/processed/graph/"
    "feature_statistics.json"
)

print("\nDone.")
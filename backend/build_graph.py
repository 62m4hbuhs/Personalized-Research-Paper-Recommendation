import os
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

GRAPH_NODES_FILE = (
    "data/processed/graph/node_mapping.csv"
)

ORIGINAL_EMBEDDINGS = (
    "data/processed/embeddings/scibert_embeddings.npy"
)

ORIGINAL_IDS = (
    "data/processed/embeddings/paper_ids.json"
)

FALLBACK_EMBEDDINGS = (
    "data/processed/graph/fallback_embeddings.npy"
)

FALLBACK_IDS = (
    "data/processed/graph/fallback_paper_ids.json"
)

OUTPUT_FEATURES = (
    "data/processed/graph/node_features.npy"
)

OUTPUT_MAPPING = (
    "data/processed/graph/final_feature_mapping.csv"
)

OUTPUT_STATS = (
    "data/processed/graph/final_feature_statistics.json"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL GRAPH FEATURE CONSTRUCTION")
print("=" * 70)


# ============================================================
# GRAPH NODES
# ============================================================

print("\nLoading graph nodes...")

nodes = pd.read_csv(
    GRAPH_NODES_FILE
)

nodes = nodes.sort_values(
    "node_id"
).reset_index(drop=True)

print(
    f"Graph nodes: {len(nodes):,}"
)


# ============================================================
# VALIDATE NODE IDS
# ============================================================

if nodes["node_id"].duplicated().any():
    raise ValueError(
        "Duplicate node_id values detected."
    )

if nodes["paperId"].duplicated().any():
    raise ValueError(
        "Duplicate paperId values detected."
    )

expected_node_ids = np.arange(
    len(nodes)
)

actual_node_ids = nodes[
    "node_id"
].to_numpy()

if not np.array_equal(
    actual_node_ids,
    expected_node_ids
):
    raise ValueError(
        "node_id values are not contiguous "
        "from 0 to N-1."
    )


# ============================================================
# LOAD ORIGINAL SCIBERT EMBEDDINGS
# ============================================================

print("\nLoading original SciBERT embeddings...")

original_embeddings = np.load(
    ORIGINAL_EMBEDDINGS,
    mmap_mode="r"
)

with open(
    ORIGINAL_IDS,
    "r",
    encoding="utf-8"
) as f:

    original_ids = json.load(f)

print(
    f"Original embeddings: "
    f"{len(original_ids):,}"
)

print(
    f"Shape: "
    f"{original_embeddings.shape}"
)


# ============================================================
# LOAD FALLBACK EMBEDDINGS
# ============================================================

print("\nLoading fallback title embeddings...")

fallback_embeddings = np.load(
    FALLBACK_EMBEDDINGS,
    mmap_mode="r"
)

with open(
    FALLBACK_IDS,
    "r",
    encoding="utf-8"
) as f:

    fallback_ids = json.load(f)

print(
    f"Fallback embeddings: "
    f"{len(fallback_ids):,}"
)

print(
    f"Shape: "
    f"{fallback_embeddings.shape}"
)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

if len(original_ids) != len(
    original_embeddings
):
    raise ValueError(
        "Original embedding ID count does not "
        "match embedding count."
    )

if len(fallback_ids) != len(
    fallback_embeddings
):
    raise ValueError(
        "Fallback embedding ID count does not "
        "match embedding count."
    )

if len(set(original_ids)) != len(
    original_ids
):
    raise ValueError(
        "Duplicate original embedding IDs."
    )

if len(set(fallback_ids)) != len(
    fallback_ids
):
    raise ValueError(
        "Duplicate fallback embedding IDs."
    )

if set(original_ids) & set(fallback_ids):
    raise ValueError(
        "Some paper IDs appear in both "
        "original and fallback embeddings."
    )


# ============================================================
# CREATE LOOKUPS
# ============================================================

print("\nCreating embedding lookups...")

original_lookup = {
    paper_id: idx
    for idx, paper_id in enumerate(
        original_ids
    )
}

fallback_lookup = {
    paper_id: idx
    for idx, paper_id in enumerate(
        fallback_ids
    )
}


# ============================================================
# DETERMINE FEATURE SOURCE
# ============================================================

print("\nDetermining feature source for each node...")

feature_source = []

original_rows = []
fallback_rows = []

for paper_id in nodes["paperId"]:

    if paper_id in original_lookup:

        feature_source.append(
            "abstract"
        )

        original_rows.append(
            original_lookup[paper_id]
        )

        fallback_rows.append(
            -1
        )

    elif paper_id in fallback_lookup:

        feature_source.append(
            "title"
        )

        original_rows.append(
            -1
        )

        fallback_rows.append(
            fallback_lookup[paper_id]
        )

    else:

        feature_source.append(
            "missing"
        )

        original_rows.append(
            -1
        )

        fallback_rows.append(
            -1
        )


nodes["feature_source"] = feature_source
nodes["original_embedding_row"] = (
    original_rows
)
nodes["fallback_embedding_row"] = (
    fallback_rows
)


# ============================================================
# COVERAGE
# ============================================================

abstract_count = (
    nodes["feature_source"]
    == "abstract"
).sum()

title_count = (
    nodes["feature_source"]
    == "title"
).sum()

missing_count = (
    nodes["feature_source"]
    == "missing"
).sum()

total = len(nodes)

print("\n" + "=" * 70)
print("FEATURE COVERAGE")
print("=" * 70)

print(
    f"Total graph nodes:       {total:,}"
)

print(
    f"Abstract features:       {abstract_count:,}"
)

print(
    f"Title fallback features: {title_count:,}"
)

print(
    f"Missing features:        {missing_count:,}"
)

print(
    f"Total feature coverage:  "
    f"{(abstract_count + title_count) / total * 100:.2f}%"
)


# ============================================================
# BUILD FINAL MATRIX
# ============================================================

dimension = original_embeddings.shape[1]

print(
    f"\nCreating final matrix: "
    f"({total:,}, {dimension})"
)

node_features = np.zeros(
    (total, dimension),
    dtype=np.float32
)


# ============================================================
# COPY ORIGINAL ABSTRACT FEATURES
# ============================================================

abstract_mask = (
    nodes["feature_source"]
    == "abstract"
)

abstract_node_ids = nodes.loc[
    abstract_mask,
    "node_id"
].to_numpy(dtype=np.int64)

abstract_embedding_rows = nodes.loc[
    abstract_mask,
    "original_embedding_row"
].to_numpy(dtype=np.int64)

print(
    f"\nCopying abstract embeddings: "
    f"{len(abstract_node_ids):,}"
)

node_features[
    abstract_node_ids
] = original_embeddings[
    abstract_embedding_rows
]


# ============================================================
# COPY FALLBACK TITLE FEATURES
# ============================================================

title_mask = (
    nodes["feature_source"]
    == "title"
)

title_node_ids = nodes.loc[
    title_mask,
    "node_id"
].to_numpy(dtype=np.int64)

title_embedding_rows = nodes.loc[
    title_mask,
    "fallback_embedding_row"
].to_numpy(dtype=np.int64)

print(
    f"Copying title embeddings: "
    f"{len(title_node_ids):,}"
)

node_features[
    title_node_ids
] = fallback_embeddings[
    title_embedding_rows
]


# ============================================================
# VALIDATION
# ============================================================

print("\nRunning validation...")

if node_features.shape != (
    total,
    dimension
):
    raise ValueError(
        "Incorrect final feature shape."
    )

if not np.isfinite(
    node_features
).all():
    raise ValueError(
        "NaN or Inf detected."
    )


# Check abstract samples
if len(abstract_node_ids) > 0:

    sample_n = min(
        100,
        len(abstract_node_ids)
    )

    assert np.allclose(
        node_features[
            abstract_node_ids[:sample_n]
        ],
        original_embeddings[
            abstract_embedding_rows[:sample_n]
        ]
    )

# Check title samples
if len(title_node_ids) > 0:

    sample_n = min(
        100,
        len(title_node_ids)
    )

    assert np.allclose(
        node_features[
            title_node_ids[:sample_n]
        ],
        fallback_embeddings[
            title_embedding_rows[:sample_n]
        ]
    )

print(
    "Feature alignment checks: PASSED"
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving final node features...")

np.save(
    OUTPUT_FEATURES,
    node_features
)

nodes[
    [
        "node_id",
        "paperId",
        "feature_source",
        "original_embedding_row",
        "fallback_embedding_row"
    ]
].to_csv(
    OUTPUT_MAPPING,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

statistics = {
    "total_graph_nodes": int(total),
    "abstract_features": int(
        abstract_count
    ),
    "title_fallback_features": int(
        title_count
    ),
    "missing_features": int(
        missing_count
    ),
    "coverage_percent": round(
        (
            abstract_count +
            title_count
        ) / total * 100,
        4
    ),
    "feature_dimension": int(
        dimension
    ),
    "dtype": "float32",
    "feature_sources": {
        "abstract": int(abstract_count),
        "title": int(title_count),
        "missing": int(missing_count)
    }
}

with open(
    OUTPUT_STATS,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL GRAPH FEATURES READY")
print("=" * 70)

print(
    f"Feature matrix: "
    f"{node_features.shape}"
)

print(
    f"Abstract features: "
    f"{abstract_count:,}"
)

print(
    f"Title features: "
    f"{title_count:,}"
)

print(
    f"Missing features: "
    f"{missing_count:,}"
)

print(
    f"Coverage: "
    f"{(abstract_count + title_count) / total * 100:.2f}%"
)

print("\nSaved:")

print(
    f" - {OUTPUT_FEATURES}"
)

print(
    f" - {OUTPUT_MAPPING}"
)

print(
    f" - {OUTPUT_STATS}"
)

print("\nDone.")
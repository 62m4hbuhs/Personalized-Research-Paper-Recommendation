# 🔍 SciBERT & FAISS Vector Search

## 1. Why SciBERT?
Standard BERT models (`bert-base-uncased`) are pretrained on Wikipedia and BookCorpus, failing on scientific vocabulary (e.g., *"inductive bias"*, *"stochastic gradient descent"*, *"homophilic graph"*, *"spectral radius"*). 

`allenai/scibert_scivocab_uncased` is pretrained specifically on 1.14M papers from Semantic Scholar (CS and Biomedicine) with a custom 30k token scientific vocabulary (`scivocab`), producing significantly more discriminative embeddings for scientific literature retrieval.

## 2. Pooling & L2 Normalization
For each paper abstract:
1. Tokenize text with truncation up to $L = 256$ tokens.
2. Pass through SciBERT encoder to obtain hidden state $H \in \mathbb{R}^{B \times L \times 768}$.
3. Apply **Mean Pooling** over active attention tokens:
   $$\vec{e} = \frac{\sum_{i=1}^{L} m_i \cdot H_i}{\sum_{i=1}^{L} m_i}$$
   where $m_i \in \{0, 1\}$ is the attention mask.
4. L2-normalize vector:
   $$\hat{e} = \frac{\vec{e}}{\|\vec{e}\|_2}$$

## 3. FAISS Indexing (`IndexFlatIP`)
- Because all vectors are L2-normalized, the inner product between query $\hat{q}$ and document $\hat{d}$ equals the cosine similarity:
  $$\langle \hat{q}, \hat{d} \rangle = \cos(\theta_{\hat{q}, \hat{d}}) \in [-1, 1]$$
- `IndexFlatIP` provides exact $k$-nearest neighbor search without quantization loss.
- Search speed over 56k vectors is $< 3\text{ ms}$, effortlessly enabling interactive sub-second user queries.

## 4. Query Handling
The system supports two query modes:
1. **Free-text Query**: Researcher types a research query (e.g., *"dynamic graph convolutional networks with temporal attention"*). Query is embedded on-the-fly and queried in FAISS.
2. **Paper-to-Paper Query**: User selects a known paper (e.g., *"How Powerful are Graph Neural Networks?"*). The system retrieves the paper's pre-computed embedding from memory (or FAISS index) and retrieves nearest neighbors, masking out the query paper itself.

---
Back to [[Dataset Specifications]] | Next: [[Citation Graph & Fallback Embeddings]]


# 💻 React Frontend Architecture

## 1. Overview & UI Goals
The user interface provides an academic discovery experience that makes the dual-signal recommendation engine tangible and interactive.

## 2. Key UI Components

### 1. Unified Search & Seed Selector
- Free-text query input box with autocomplete.
- "Select Seed Paper" mode: search and pick any paper already in the dataset to discover its citation & semantic neighborhood.

### 2. Multi-Signal Weighting Slider
- Interactive slider for the hyperparameter $\alpha \in [0, 1]$:
  - Slide to left (0%): Citation Graph Only
  - Slide to center (60%): Balanced Hybrid (Default)
  - Slide to right (100%): Semantic Content Only
- Instantly re-ranks the candidates dynamically without re-querying the heavy FAISS index.

### 3. Paper Recommendation Cards
Each card displays:
- Rank badge (`#1`, `#2`, etc.)
- Paper Title & Year
- Authors & Venue
- Citation count badge
- Hybrid Relevance Score (%) with animated progress bar
- Decomposed Score Pill:
  - Semantic Match (`88%`)
  - Citation Graph Proximity (`92%`)
- "Why recommended?" collapsible explanation pill.
- Direct external link (arXiv / Semantic Scholar / DOI).

### 4. Interactive Citation Subgraph Explorer
- Visualizes 1-hop and 2-hop ego network connecting the seed paper to recommended candidates using SVG / Canvas or Force-Directed layout.

### 5. Evaluation & Experiment Tab
- A live evaluation dashboard showing benchmark comparisons:
  - Semantic-Only vs. Hybrid GraphSAGE
  - ROC curve, MRR, Hits@5, Hits@10 metrics table.

---
Back to [[Index]] | Next: [[Leakage-Safe Evaluation Protocol]]


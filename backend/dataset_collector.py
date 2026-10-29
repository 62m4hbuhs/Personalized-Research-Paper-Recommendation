import os
import json
import csv
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

if not API_KEY:
    raise ValueError(
        "SEMANTIC_SCHOLAR_API_KEY not found in .env"
    )

BASE_URL = "https://api.semanticscholar.org/graph/v1"


# ------------------------------------------------------------
# DATA DIRECTORIES
# ------------------------------------------------------------

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_FILE = OUTPUT_DIR / "papers.jsonl"
CITATIONS_FILE = OUTPUT_DIR / "citations.csv"


# ------------------------------------------------------------
# DATASET SETTINGS
# ------------------------------------------------------------

TARGET_PAPERS = 5000

# Semantic Scholar bulk search limit
BULK_LIMIT = 1000


# ------------------------------------------------------------
# SEARCH TOPICS
# ------------------------------------------------------------

SEARCH_QUERIES = [
    "machine learning",
    "deep learning",
    "graph neural networks",
    "natural language processing",
    "large language models",
    "recommender systems",
    "information retrieval",
    "knowledge graphs",
    "representation learning",
    "AI agents",
]


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "x-api-key": API_KEY,
    "User-Agent": "MajorProject-ResearchPaperRecommender/1.0"
})


# ============================================================
# HELPER — GET REQUEST WITH RETRIES
# ============================================================

def api_get(url, params=None, retries=6):
    """
    Make a GET request with retry and exponential backoff.
    """

    for attempt in range(retries):

        try:

            response = session.get(
                url,
                params=params,
                timeout=60
            )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if response.status_code == 200:
                return response.json()

            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------

            if response.status_code == 429:

                wait_time = min(
                    60,
                    2 ** attempt
                )

                print(
                    f"\nRate limit reached. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            # ------------------------------------------------
            # Server error
            # ------------------------------------------------

            if response.status_code >= 500:

                wait_time = min(
                    60,
                    2 ** attempt
                )

                print(
                    f"\nServer error "
                    f"{response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            # ------------------------------------------------
            # Authentication
            # ------------------------------------------------

            if response.status_code == 401:

                raise RuntimeError(
                    "401 Unauthorized: "
                    "Check your Semantic Scholar API key."
                )

            if response.status_code == 403:

                raise RuntimeError(
                    "403 Forbidden: "
                    "API access was denied."
                )

            # ------------------------------------------------
            # Other error
            # ------------------------------------------------

            print("\nAPI ERROR")
            print("Status:", response.status_code)
            print("Response:", response.text[:500])

            return None

        except requests.exceptions.RequestException as e:

            wait_time = min(
                60,
                2 ** attempt
            )

            print(
                f"\nNetwork error: {e}"
            )

            print(
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    return None


# ============================================================
# STEP 1 — COLLECT PAPER METADATA
# ============================================================

def collect_papers():

    print("\n" + "=" * 70)
    print("STEP 1 — COLLECTING RESEARCH PAPERS")
    print("=" * 70)

    papers = {}

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if PAPERS_FILE.exists():

        print(
            f"Existing dataset found: {PAPERS_FILE.resolve()}"
        )

        with open(
            PAPERS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                try:

                    paper = json.loads(line)

                    if paper.get("paperId"):
                        papers[paper["paperId"]] = paper

                except json.JSONDecodeError:
                    continue

        print(
            f"Loaded {len(papers)} existing papers."
        )

    # --------------------------------------------------------
    # Search topics
    # --------------------------------------------------------

    for query in SEARCH_QUERIES:

        if len(papers) >= TARGET_PAPERS:
            break

        print(
            f"\nSearching: {query}"
        )

        token = None

        while len(papers) < TARGET_PAPERS:

            params = {
                "query": query,

                "fields": (
                    "paperId,"
                    "title,"
                    "abstract,"
                    "authors,"
                    "year,"
                    "venue,"
                    "citationCount,"
                    "fieldsOfStudy,"
                    "publicationDate,"
                    "openAccessPdf"
                ),

                "limit": BULK_LIMIT,

                "fieldsOfStudy": "Computer Science"
            }

            if token:
                params["token"] = token

            result = api_get(
                f"{BASE_URL}/paper/search/bulk",
                params=params
            )

            if result is None:

                print(
                    "Could not retrieve this batch."
                )

                break

            batch = result.get("data", [])

            if not batch:

                print(
                    "No more papers for this query."
                )

                break

            added = 0

            for paper in batch:

                paper_id = paper.get("paperId")

                if not paper_id:
                    continue

                # ------------------------------------------------
                # Require title
                # ------------------------------------------------

                title = paper.get("title")

                if not title:
                    continue

                # ------------------------------------------------
                # Require abstract
                # ------------------------------------------------

                abstract = paper.get("abstract")

                if not abstract:
                    continue

                # ------------------------------------------------
                # Avoid duplicates
                # ------------------------------------------------

                if paper_id in papers:
                    continue

                # ------------------------------------------------
                # Clean record
                # ------------------------------------------------

                cleaned = {
                    "paperId": paper_id,

                    "title": title.strip(),

                    "abstract": abstract.strip(),

                    "authors": [
                        {
                            "authorId": author.get("authorId"),
                            "name": author.get("name")
                        }
                        for author in paper.get(
                            "authors",
                            []
                        )
                    ],

                    "year": paper.get("year"),

                    "venue": paper.get("venue"),

                    "citationCount": paper.get(
                        "citationCount",
                        0
                    ),

                    "fieldsOfStudy": paper.get(
                        "fieldsOfStudy",
                        []
                    ),

                    "publicationDate": paper.get(
                        "publicationDate"
                    ),

                    "openAccessPdf": paper.get(
                        "openAccessPdf"
                    )
                }

                papers[paper_id] = cleaned

                added += 1

                if len(papers) >= TARGET_PAPERS:
                    break

            print(
                f"Received: {len(batch):4d} | "
                f"New: {added:4d} | "
                f"Total: {len(papers):5d}"
            )

            # ------------------------------------------------
            # Next token
            # ------------------------------------------------

            token = result.get("token")

            if not token:
                break

            # Small delay
            time.sleep(1.1)

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    print(
        f"\nSaving {len(papers)} papers..."
    )

    with open(
        PAPERS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for paper in papers.values():

            f.write(
                json.dumps(
                    paper,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Saved to: {PAPERS_FILE.resolve()}"
    )

    return papers


# ============================================================
# STEP 2 — COLLECT CITATION / REFERENCE GRAPH
# ============================================================

def collect_citations(papers):

    print("\n" + "=" * 70)
    print("STEP 2 — BUILDING CITATION GRAPH")
    print("=" * 70)

    paper_ids = list(papers.keys())

    print(
        f"Total papers to process: {len(paper_ids)}"
    )

    print(
        f"Citation file: {CITATIONS_FILE.resolve()}"
    )

    # --------------------------------------------------------
    # Existing edges for resume support
    # --------------------------------------------------------

    existing_edges = set()

    if CITATIONS_FILE.exists():

        print(
            f"\nExisting citation file found:"
        )

        print(
            CITATIONS_FILE.resolve()
        )

        with open(
            CITATIONS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                source = row.get("source_paper")
                target = row.get("target_paper")

                if source and target:

                    existing_edges.add(
                        (
                            source,
                            target
                        )
                    )

        print(
            f"Loaded {len(existing_edges)} existing edges."
        )

    else:

        print(
            "\nNo existing citation file found."
        )

        print(
            "A new one will be created."
        )

    # --------------------------------------------------------
    # Open CSV
    # --------------------------------------------------------

    file_exists = CITATIONS_FILE.exists()

    with open(
        CITATIONS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        # ----------------------------------------------------
        # CSV header
        # ----------------------------------------------------

        if not file_exists:

            writer.writerow([
                "source_paper",
                "target_paper",
                "relationship"
            ])

        # ----------------------------------------------------
        # IMPORTANT:
        # Use batch requests instead of 1 request per paper.
        # ----------------------------------------------------

        BATCH_SIZE = 100

        total_batches = (
            len(paper_ids) + BATCH_SIZE - 1
        ) // BATCH_SIZE

        print(
            f"\nBatch size: {BATCH_SIZE}"
        )

        print(
            f"Total API batches: {total_batches}"
        )

        # ----------------------------------------------------
        # Process batches
        # ----------------------------------------------------

        for batch_number in range(total_batches):

            start = batch_number * BATCH_SIZE

            end = min(
                start + BATCH_SIZE,
                len(paper_ids)
            )

            batch_ids = paper_ids[start:end]

            print(
                f"\n{'-' * 70}"
            )

            print(
                f"Batch {batch_number + 1}/"
                f"{total_batches}"
            )

            print(
                f"Papers {start + 1}-{end}"
            )

            # ------------------------------------------------
            # API request
            # ------------------------------------------------

            url = f"{BASE_URL}/paper/batch"

            params = {
                "fields": (
                    "paperId,"
                    "references.paperId"
                )
            }

            response = None

            # ------------------------------------------------
            # Retry mechanism
            # ------------------------------------------------

            for attempt in range(6):

                try:

                    response = session.post(
                        url,
                        params=params,
                        json={
                            "ids": batch_ids
                        },
                        timeout=120
                    )

                    # ----------------------------------------
                    # Success
                    # ----------------------------------------

                    if response.status_code == 200:
                        break

                    # ----------------------------------------
                    # Rate limit
                    # ----------------------------------------

                    if response.status_code == 429:

                        wait = min(
                            60,
                            2 ** attempt
                        )

                        print(
                            f"Rate limit reached. "
                            f"Waiting {wait}s..."
                        )

                        time.sleep(wait)

                        continue

                    # ----------------------------------------
                    # Server error
                    # ----------------------------------------

                    if response.status_code >= 500:

                        wait = min(
                            60,
                            2 ** attempt
                        )

                        print(
                            f"Server error "
                            f"{response.status_code}. "
                            f"Waiting {wait}s..."
                        )

                        time.sleep(wait)

                        continue

                    # ----------------------------------------
                    # Other API error
                    # ----------------------------------------

                    print(
                        f"API error "
                        f"{response.status_code}:"
                    )

                    print(
                        response.text[:500]
                    )

                    break

                except requests.exceptions.RequestException as e:

                    wait = min(
                        60,
                        2 ** attempt
                    )

                    print(
                        f"Network error: {e}"
                    )

                    print(
                        f"Waiting {wait}s..."
                    )

                    time.sleep(wait)

            # ------------------------------------------------
            # Check response
            # ------------------------------------------------

            if response is None:

                print(
                    "No response received."
                )

                continue

            if response.status_code != 200:

                print(
                    f"Skipping batch."
                )

                print(
                    f"Status: {response.status_code}"
                )

                continue

            # ------------------------------------------------
            # Parse response
            # ------------------------------------------------

            try:

                results = response.json()

            except Exception as e:

                print(
                    "Could not parse API response."
                )

                print(
                    f"Error: {e}"
                )

                continue

            if not isinstance(results, list):

                print(
                    "Unexpected API response format."
                )

                continue

            # ------------------------------------------------
            # Process papers
            # ------------------------------------------------

            batch_edges = 0

            returned_papers = 0

            for paper in results:

                if not paper:
                    continue

                returned_papers += 1

                source_id = paper.get(
                    "paperId"
                )

                if not source_id:
                    continue

                # ============================================
                # REFERENCES
                #
                # source paper → referenced paper
                #
                # Example:
                #
                # Paper A references Paper B
                #
                # A → B
                # ============================================

                references = paper.get(
                    "references"
                ) or []

                for reference in references:

                    target_id = reference.get(
                        "paperId"
                    )

                    if not target_id:
                        continue

                    edge = (
                        source_id,
                        target_id
                    )

                    # Avoid duplicate edge
                    if edge in existing_edges:
                        continue

                    writer.writerow([
                        source_id,
                        target_id,
                        "reference"
                    ])

                    existing_edges.add(edge)

                    batch_edges += 1

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            f.flush()

            print(
                f"Returned papers: "
                f"{returned_papers}"
            )

            print(
                f"New graph edges: "
                f"{batch_edges}"
            )

            print(
                f"Total graph edges: "
                f"{len(existing_edges)}"
            )

            # Small delay between batches
            time.sleep(1.1)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CITATION GRAPH COMPLETE")
    print("=" * 70)

    print(
        f"Total graph edges: "
        f"{len(existing_edges)}"
    )

    print(
        f"Saved to:"
    )

    print(
        f"{CITATIONS_FILE.resolve()}"
    )


# ============================================================
# STEP 3 — DATASET SUMMARY
# ============================================================

def print_summary(papers):

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Number of papers: {len(papers)}"
    )

    # --------------------------------------------------------
    # Abstract count
    # --------------------------------------------------------

    abstracts = sum(
        1
        for paper in papers.values()
        if paper.get("abstract")
    )

    print(
        f"Papers with abstracts: {abstracts}"
    )

    # --------------------------------------------------------
    # Year range
    # --------------------------------------------------------

    years = [
        paper["year"]
        for paper in papers.values()
        if paper.get("year")
    ]

    if years:

        print(
            f"Year range: "
            f"{min(years)} - {max(years)}"
        )

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    print(
        "\nPaper file:"
    )

    print(
        f"  {PAPERS_FILE.resolve()}"
    )

    print(
        "\nCitation file:"
    )

    print(
        f"  {CITATIONS_FILE.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("PERSONALIZED RESEARCH PAPER RECOMMENDER")
    print("Semantic Scholar Dataset Collector")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    papers = collect_papers()

    if not papers:

        print(
            "\nNo papers collected."
        )

        return

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    collect_citations(papers)

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print_summary(papers)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\nDataset collection completed!"
    )

    print(
        "\nNext steps:"
    )

    print(
        "1. Clean and validate papers"
    )

    print(
        "2. Generate SciBERT embeddings"
    )

    print(
        "3. Build FAISS index"
    )

    print(
        "4. Build citation graph"
    )

    print(
        "5. Train GraphSAGE"
    )

    print(
        "6. Implement hybrid ranking"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
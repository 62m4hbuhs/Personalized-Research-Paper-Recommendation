import json
import os
import time
import random
import csv
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

PAPERS_FILE = DATA_DIR / "papers.jsonl"
CITATIONS_FILE = DATA_DIR / "citations.csv"
OUTPUT_FILE = DATA_DIR / "connected_papers.jsonl"
FAILED_FILE = DATA_DIR / "failed_paper_ids.txt"


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

# Semantic Scholar maximum
BATCH_SIZE = 500

# API key introductory limit = 1 request/sec
# We use 1.5 sec to stay safely below the limit.
REQUEST_INTERVAL = 1.5

MAX_RETRIES = 8

FIELDS = (
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
)


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "SEMANTIC_SCHOLAR_API_KEY not found in .env"
    )


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "x-api-key": API_KEY,
    "User-Agent": "MajorProject-ResearchPaperRecommender/1.0",
    "Accept": "application/json"
})


# ============================================================
# LOAD ORIGINAL PAPER IDS
# ============================================================

def load_original_ids():

    ids = set()

    print("Loading original papers...")

    with open(
        PAPERS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                paper = json.loads(line)

                paper_id = paper.get("paperId")

                if paper_id:
                    ids.add(paper_id)

            except json.JSONDecodeError:
                pass

    return ids


# ============================================================
# LOAD CITATION GRAPH IDS
# ============================================================

def load_graph_ids():

    ids = set()

    print("Loading citation graph IDs...")

    with open(
        CITATIONS_FILE,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            source = row.get("source_paper")
            target = row.get("target_paper")

            if source:
                ids.add(source.strip())

            if target:
                ids.add(target.strip())

    return ids


# ============================================================
# LOAD ALREADY DOWNLOADED
# ============================================================

def load_downloaded_ids():

    ids = set()

    if not OUTPUT_FILE.exists():
        return ids

    print("Checking already downloaded papers...")

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                paper = json.loads(line)

                paper_id = paper.get("paperId")

                if paper_id:
                    ids.add(paper_id)

            except json.JSONDecodeError:
                pass

    return ids


# ============================================================
# LOAD FAILED IDS
# ============================================================

def load_failed_ids():

    ids = set()

    if not FAILED_FILE.exists():
        return ids

    with open(
        FAILED_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            paper_id = line.strip()

            if paper_id:
                ids.add(paper_id)

    return ids


# ============================================================
# SAVE FAILED ID
# ============================================================

def save_failed_ids(ids):

    if not ids:
        return

    with open(
        FAILED_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        for paper_id in ids:
            f.write(paper_id + "\n")


# ============================================================
# FETCH BATCH
# ============================================================

def fetch_batch(paper_ids):

    payload = {
        "ids": paper_ids
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.post(
                API_URL,
                params={"fields": FIELDS},
                json=payload,
                timeout=120
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                data = response.json()

                if not isinstance(data, list):
                    print("Unexpected response format.")
                    return None

                return data


            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 10

                else:

                    # Conservative exponential backoff
                    wait = min(
                        10 * (2 ** (attempt - 1)),
                        120
                    )

                wait += random.uniform(1, 3)

                print(
                    f"429 RATE LIMITED | "
                    f"attempt {attempt}/{MAX_RETRIES} | "
                    f"waiting {wait:.1f}s"
                )

                time.sleep(wait)

                continue


            # ------------------------------------------------
            # SERVER ERROR
            # ------------------------------------------------

            if response.status_code >= 500:

                wait = min(
                    10 * attempt,
                    120
                )

                print(
                    f"{response.status_code} SERVER ERROR | "
                    f"waiting {wait}s"
                )

                time.sleep(wait)

                continue


            # ------------------------------------------------
            # AUTHENTICATION
            # ------------------------------------------------

            if response.status_code in (401, 403):

                print()
                print("=" * 60)
                print("API KEY / PERMISSION ERROR")
                print("=" * 60)
                print("Status:", response.status_code)
                print(response.text[:1000])

                return None


            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            print(
                f"API ERROR {response.status_code}: "
                f"{response.text[:500]}"
            )

            return None


        except requests.exceptions.Timeout:

            wait = min(
                10 * attempt,
                120
            )

            print(
                f"TIMEOUT | waiting {wait}s"
            )

            time.sleep(wait)


        except requests.exceptions.RequestException as e:

            wait = min(
                10 * attempt,
                120
            )

            print(
                f"NETWORK ERROR: {e}"
            )

            time.sleep(wait)


    print(
        "Batch failed after maximum retries."
    )

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SEMANTIC SCHOLAR DATASET EXPANSION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    original_ids = load_original_ids()

    print(
        f"Original papers: "
        f"{len(original_ids):,}"
    )

    graph_ids = load_graph_ids()

    print(
        f"Unique papers in citation graph: "
        f"{len(graph_ids):,}"
    )

    downloaded_ids = load_downloaded_ids()

    print(
        f"Already downloaded: "
        f"{len(downloaded_ids):,}"
    )

    failed_ids = load_failed_ids()

    print(
        f"Previously failed: "
        f"{len(failed_ids):,}"
    )

    # --------------------------------------------------------
    # Determine remaining IDs
    # --------------------------------------------------------

    ids_to_fetch = (
        graph_ids
        - original_ids
        - downloaded_ids
        - failed_ids
    )

    ids_to_fetch = sorted(ids_to_fetch)

    print()
    print(
        f"Remaining papers to fetch: "
        f"{len(ids_to_fetch):,}"
    )

    if not ids_to_fetch:

        print()
        print("Nothing left to download.")
        return

    # --------------------------------------------------------
    # Calculate batches
    # --------------------------------------------------------

    total_batches = (
        len(ids_to_fetch) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print()
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Request interval: {REQUEST_INTERVAL}s")
    print(f"Total batches:    {total_batches:,}")
    print()

    # --------------------------------------------------------
    # Make sure output directory exists
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    total_saved = 0
    total_missing = 0

    for batch_index, start in enumerate(
        range(
            0,
            len(ids_to_fetch),
            BATCH_SIZE
        ),
        start=1
    ):

        batch_ids = ids_to_fetch[
            start:start + BATCH_SIZE
        ]

        print()
        print(
            f"Batch {batch_index}/{total_batches} "
            f"({len(batch_ids)} papers)"
        )

        results = fetch_batch(batch_ids)

        # ----------------------------------------------------
        # Stop on serious failure
        # ----------------------------------------------------

        if results is None:

            print()
            print(
                "Batch could not be completed."
            )

            print(
                "Stopping safely."
            )

            print(
                "Run the script again later; "
                "already saved papers will be skipped."
            )

            return

        # ----------------------------------------------------
        # Save returned papers
        # ----------------------------------------------------

        returned_ids = set()

        with open(
            OUTPUT_FILE,
            "a",
            encoding="utf-8"
        ) as out:

            for paper in results:

                if not paper:
                    continue

                paper_id = paper.get("paperId")

                if not paper_id:
                    continue

                returned_ids.add(paper_id)

                out.write(
                    json.dumps(
                        paper,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                total_saved += 1

        # ----------------------------------------------------
        # Find IDs API didn't return
        # ----------------------------------------------------

        missing_ids = (
            set(batch_ids)
            - returned_ids
        )

        if missing_ids:

            print(
                f"Missing/unavailable IDs: "
                f"{len(missing_ids)}"
            )

            save_failed_ids(
                missing_ids
            )

            total_missing += len(
                missing_ids
            )

        else:

            print(
                "All IDs returned successfully."
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = min(
            start + len(batch_ids),
            len(ids_to_fetch)
        )

        percent = (
            processed /
            len(ids_to_fetch)
        ) * 100

        print(
            f"Progress: "
            f"{processed:,}/"
            f"{len(ids_to_fetch):,} "
            f"({percent:.2f}%)"
        )

        print(
            f"Total saved this run: "
            f"{total_saved:,}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Wait before next request
        # ----------------------------------------------------

        if processed < len(ids_to_fetch):

            time.sleep(
                REQUEST_INTERVAL
                + random.uniform(0.2, 0.8)
            )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPANSION COMPLETE")
    print("=" * 70)

    print(
        f"Saved this run:      {total_saved:,}"
    )

    print(
        f"Missing/unavailable: {total_missing:,}"
    )

    print(
        f"Output file:         {OUTPUT_FILE}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
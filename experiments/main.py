import os
import requests
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Get Semantic Scholar API key
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

if not API_KEY:
    raise ValueError(
        "SEMANTIC_SCHOLAR_API_KEY not found. "
        "Make sure it is present in your .env file."
    )

# Semantic Scholar API endpoint
URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Request parameters
params = {
    "query": "graph neural networks",
    "limit": 5,
    "fields": "paperId,title,abstract,authors,year,venue,citationCount"
}

# API authentication
headers = {
    "x-api-key": API_KEY
}


def test_api():
    print("=" * 60)
    print("Testing Semantic Scholar API")
    print("=" * 60)

    try:
        response = requests.get(
            URL,
            params=params,
            headers=headers,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("\nAPI KEY WORKING!")
            print("\nPapers returned:\n")

            data = response.json()

            print(f"Total matching papers: {data.get('total')}")

            for i, paper in enumerate(data.get("data", []), start=1):
                print(f"\n{i}. {paper.get('title')}")
                print(f"   Paper ID: {paper.get('paperId')}")
                print(f"   Year: {paper.get('year')}")
                print(f"   Venue: {paper.get('venue')}")
                print(f"   Citations: {paper.get('citationCount')}")

                authors = paper.get("authors", [])
                author_names = [
                    author.get("name")
                    for author in authors
                    if author.get("name")
                ]

                print(f"   Authors: {', '.join(author_names)}")

                abstract = paper.get("abstract")

                if abstract:
                    print(f"   Abstract: {abstract[:200]}...")
                else:
                    print("   Abstract: Not available")

        elif response.status_code == 401:
            print("\nERROR 401: Invalid or missing API key.")

        elif response.status_code == 403:
            print("\nERROR 403: Access forbidden.")

        elif response.status_code == 429:
            print("\nERROR 429: Rate limit reached.")

        else:
            print("\nAPI request failed.")
            print(response.text)

    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out.")

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    test_api()

import json
from pathlib import Path

from minsearch import Index

BASE_DIR = Path(__file__).resolve().parent


def load_currency_data(file_path=None):
    documents = []

    if file_path is None:
        file_path = BASE_DIR / "documents" / "currencies.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support your JSON structure: {"currencies": [...]}
    if isinstance(data, dict):
        documents = data.get("currencies")

        if not isinstance(documents, list):
            raise ValueError(
                "Expected 'currencies' to contain a JSON list."
            )
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError(
            "Expected currencies.json to contain a JSON list "
            "or an object with a 'currencies' key."
        )

    # Validate and prepare documents
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(
                f"Currency at index {i} must be a JSON object."
            )

        required_fields = ["code", "name", "symbol"]

        for field in required_fields:
            if field not in doc:
                raise ValueError(
                    f"Currency at index {i} is missing '{field}'."
                )

        # Add a unique ID for database storage
        doc["doc_id"] = doc["code"]

        # Convert the countries list to searchable text
        doc["countries_text"] = ", ".join(
            doc.get("countries", [])
        )

    return documents


def build_index(documents):
    index = Index(
        text_fields=[
            "code",
            "name",
            "symbol",
            "countries_text",
        ],
        keyword_fields=[
            "code",
        ],
    )

    index.fit(documents)

    return index

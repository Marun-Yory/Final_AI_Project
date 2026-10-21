import json

from minsearch import Index


def load_currency_data():
    file_path = "./documents/currencies.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data["currencies"]

    for doc in documents:
        # Use currency code as the unique document ID
        doc["doc_id"] = doc["code"]

    return documents



def build_index(documents):
    # Convert countries lists into searchable strings
    for doc in documents:
        if isinstance(doc.get("countries"), list):
            doc["countries"] = " ".join(doc["countries"])

    index = Index(
        text_fields=["code", "name", "symbol", "countries"],
        keyword_fields=["decimal_digits"]
    )

    index.fit(documents)

    return index


def search_currencies(index, query, num_results=5):
    results = index.search(
        query,
        boost_dict={
            "code": 3,
            "name": 3,
            "symbol": 2,
            "countries": 1
        },
        num_results=num_results
    )

    return results


if __name__ == "__main__":
    documents = load_currency_data()
    index = build_index(documents)

    results = search_currencies(index, "Japanese Yen")

    for result in results:
        print(result)

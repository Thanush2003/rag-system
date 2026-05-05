import json

from pymilvus import (
    connections,
    Collection
)

from sentence_transformers import SentenceTransformer

# -----------------------------
# CONNECT TO MILVUS
# -----------------------------
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

collection = Collection("pdf_rag")
collection.load()

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD QA FILE
# -----------------------------
with open(
    "../eval/qa.jsonl",
    "r",
    encoding="utf-8"
) as f:

    samples = [
        json.loads(line)
        for line in f
    ]

# -----------------------------
# EVALUATION
# -----------------------------
correct = 0

total = len(samples)

for sample in samples:

    question = sample["question"]

    expected_source = sample["source"]

    expected_pages = str(
        sample["page"]
    ).split(",")

    print("\n====================")
    print("QUESTION:", question)

    print(
        "EXPECTED PAGES:",
        expected_pages
    )

    # Generate embedding
    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Search top 3
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=3,
        output_fields=[
            "text",
            "source",
            "page"
        ]
    )

    found = False

    print("\nTOP 3 RESULTS:")

    for hit in results[0]:

        source = hit.entity.get("source")

        page = str(
            hit.entity.get("page")
        )

        score = hit.distance

        print(
            f"\nSOURCE: {source}"
        )

        print(
            f"PAGE: {page}"
        )

        print(
            f"SCORE: {score}"
        )

        # -----------------------------
        # PAGE + SOURCE CHECK
        # -----------------------------
        if expected_source == source:

            if page in expected_pages:

                found = True

    # -----------------------------
    # RESULT
    # -----------------------------
    if found:

        correct += 1

        print("\nRESULT: PASS")

    else:

        print("\nRESULT: FAIL")

# -----------------------------
# FINAL RECALL
# -----------------------------
recall = correct / total

print("\n====================")
print(f"Recall@3: {recall:.2f}")
print("====================")
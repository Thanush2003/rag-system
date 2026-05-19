import json

from pymilvus import (
    connections,
    Collection
)

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi

import ollama

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
# BM25 INITIALIZATION
# -----------------------------
bm25_docs = collection.query(
    expr="id >= 0",
    output_fields=[
        "text",
        "parent_text",
        "source",
        "page"
    ],
    limit=16000
)

bm25_texts = [
    doc["text"]
    for doc in bm25_docs
]

tokenized_corpus = [
    text.split()
    for text in bm25_texts
]

bm25 = BM25Okapi(
    tokenized_corpus
)

print("BM25 initialized!")

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# -----------------------------
# LOAD RERANKER
# -----------------------------
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker loaded!")

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

    # -----------------------------
    # QUERY REWRITING
    # -----------------------------
    rewrite_prompt = f"""
Generate 3 different search queries
for the following question.

Question:
{question}

Return only the rewritten queries.
"""

    rewrite_response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )

    rewritten_queries = (
        rewrite_response["message"]["content"]
        .split("\n")
    )

    all_queries = [
        question
    ] + rewritten_queries

    # -----------------------------
    # MULTI QUERY DENSE RETRIEVAL
    # -----------------------------
    all_results = []

    for q in all_queries:

        query_embedding = embedding_model.encode(
            [q]
        ).tolist()

        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param={
                "metric_type": "COSINE"
            },
            limit=20,
            output_fields=[
                "text",
                "parent_text",
                "source",
                "page"
            ]
        )

        for hit in results[0]:

            all_results.append(hit)

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------
    unique_results = {}

    for hit in all_results:

        text = hit.entity.get("text")

        if text not in unique_results:

            unique_results[text] = hit

    final_results = list(
        unique_results.values()
    )

    # -----------------------------
    # BM25 RETRIEVAL
    # -----------------------------
    tokenized_query = question.split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    # -----------------------------
    # RRF FUSION
    # -----------------------------
    rrf_scores = {}

    k = 60

    # Dense ranking
    for rank, hit in enumerate(
        final_results
    ):

        text = hit.entity.get("text")

        rrf_scores[text] = (
            rrf_scores.get(text, 0)
            + 1 / (k + rank + 1)
        )

    # BM25 ranking
    for rank, idx in enumerate(
        top_bm25_indices
    ):

        text = bm25_texts[idx]

        rrf_scores[text] = (
            rrf_scores.get(text, 0)
            + 1 / (k + rank + 1)
        )

    # -----------------------------
    # FINAL SORTED RESULTS
    # -----------------------------
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------
    # CROSS ENCODER RERANKING
    # -----------------------------
    top_candidates = sorted_results[:20]

    rerank_pairs = []

    for text, score in top_candidates:

        rerank_pairs.append(
            [question, text]
        )

    rerank_scores = reranker.predict(
        rerank_pairs
    )

    reranked_results = []

    for i, (text, _) in enumerate(
        top_candidates
    ):

        reranked_results.append(
            (
                text,
                rerank_scores[i]
            )
        )

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------
    # CHECK TOP 5 RESULTS
    # -----------------------------
    found = False

    print("\nTOP 5 RESULTS:")

    for text, score in reranked_results[:5]:

        for hit in final_results:

            if hit.entity.get("text") == text:

                source = hit.entity.get(
                    "source"
                )

                page = str(
                    hit.entity.get("page")
                )

                print(
                    f"\nSOURCE: {source}"
                )

                print(
                    f"PAGE: {page}"
                )

                print(
                    f"RERANK SCORE: {score}"
                )

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

print(f"Recall@5: {recall:.2f}")

print("====================")























































































































# import json

# from pymilvus import (
#     connections,
#     Collection
# )

# from sentence_transformers import SentenceTransformer

# # -----------------------------
# # CONNECT TO MILVUS
# # -----------------------------
# connections.connect(
#     alias="default",
#     host="localhost",
#     port="19530"
# )

# collection = Collection("pdf_rag")
# collection.load()

# # -----------------------------
# # LOAD EMBEDDING MODEL
# # -----------------------------
# embedding_model = SentenceTransformer(
#     "BAAI/bge-base-en-v1.5"
# )

# # -----------------------------
# # LOAD QA FILE
# # -----------------------------
# with open(
#     "../eval/qa.jsonl",
#     "r",
#     encoding="utf-8"
# ) as f:

#     samples = [
#         json.loads(line)
#         for line in f
#     ]

# # -----------------------------
# # EVALUATION
# # -----------------------------
# correct = 0

# total = len(samples)

# for sample in samples:

#     question = sample["question"]

#     expected_source = sample["source"]

#     expected_pages = str(
#         sample["page"]
#     ).split(",")

#     print("\n====================")
#     print("QUESTION:", question)

#     print(
#         "EXPECTED PAGES:",
#         expected_pages
#     )

#     # Generate embedding
#     query_embedding = embedding_model.encode(
#         [question]
#     ).tolist()

#     # Search top 3
#     results = collection.search(
#         data=query_embedding,
#         anns_field="embedding",
#         param={"metric_type": "COSINE"},
#         limit=3,
#         output_fields=[
#             "text",
#             "source",
#             "page"
#         ]
#     )

#     found = False

#     print("\nTOP 3 RESULTS:")

#     for hit in results[0]:

#         source = hit.entity.get("source")

#         page = str(
#             hit.entity.get("page")
#         )

#         score = hit.distance

#         print(
#             f"\nSOURCE: {source}"
#         )

#         print(
#             f"PAGE: {page}"
#         )

#         print(
#             f"SCORE: {score}"
#         )

#         # -----------------------------
#         # PAGE + SOURCE CHECK
#         # -----------------------------
#         if expected_source == source:

#             if page in expected_pages:

#                 found = True

#     # -----------------------------
#     # RESULT
#     # -----------------------------
#     if found:

#         correct += 1

#         print("\nRESULT: PASS")

#     else:

#         print("\nRESULT: FAIL")

# # -----------------------------
# # FINAL RECALL
# # -----------------------------
# recall = correct / total

# print("\n====================")
# print(f"Recall@3: {recall:.2f}")
# print("====================")
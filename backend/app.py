from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from pymilvus import connections, Collection

from sentence_transformers import SentenceTransformer
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect Milvus Lite
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

# Embedding model
embedding_model = SentenceTransformer(  
    "BAAI/bge-base-en-v1.5"
)

# Reranking model
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker loaded!")
# Request schema
class Query(BaseModel):
    question: str
    history: str = ""

@app.post("/chat")
def chat(query: Query):
    # -----------------------------
    # QUERY REWRITING
    # -----------------------------
    rewrite_prompt = f"""
    Generate 3 different search queries for the following question.

    Question:
    {query.question}

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
        query.question
    ] + rewritten_queries
    print(all_queries)

    # -----------------------------
    # MULTI QUERY RETRIEVAL
    # -----------------------------
    all_results = []

    for q in all_queries:

        query_embedding = embedding_model.encode(
            [q]
        ).tolist()

        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=20,
            output_fields=["text","parent_text","source","page"]
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
    tokenized_query = (
        query.question.split()
    )

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

    # -----------------------------
    # DENSE RETRIEVAL RANKING
    # -----------------------------
    for rank, hit in enumerate(final_results):

        text = hit.entity.get("text")

        rrf_scores[text] = (
            rrf_scores.get(text, 0)
            + 1 / (k + rank + 1)
        )

    # -----------------------------
    # BM25 RETRIEVAL RANKING
    # -----------------------------
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
            [query.question, text]
        )

    rerank_scores = reranker.predict(
        rerank_pairs
    )
    reranked_results = []

    for i, (text, _) in enumerate(top_candidates):
        reranked_results.append((text,rerank_scores[i]))

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x[1],
        reverse=True
    )
    # print(top_score)

# Threshold check
    # if top_score < 0.7:

    #     def no_answer():
    #         yield (
    #             "I could not find relevant "
    #             "information in the provided documents."
    #         )

    #     return StreamingResponse(
    #         no_answer(),
    #         media_type="text/plain"
    #     )

    context = ""

    used = set()
    relevant_chunks = 0

    for text, score in reranked_results:

        for hit in final_results:

            if hit.entity.get("text") == text:

                parent_text = hit.entity.get(
                    "parent_text"
                )

                if parent_text not in used:

                    context += (
                        parent_text + "\n\n"
                    )

                    used.add(parent_text)
                    relevant_chunks += 1

                break
        if len(used) >= 5:
            break    
    top_score = reranked_results[0][1]

    # -----------------------------
    # CONTEXT VALIDATION
    # -----------------------------
    # if(len(context.strip()) < 100):

    #     def no_answer():
    #         yield (
    #             "I could not find relevant "
    #             "information in the provided documents."
    #         )

    #     return StreamingResponse(
    #         no_answer(),
    #         media_type="text/plain"
    #     )

    # -----------------------------
    # ANSWER VERIFICATION
    # -----------------------------
    verification_prompt = f"""
    Determine whether the answer to the question
    is explicitly present in the context.

    Reply ONLY with:
    YES
    or
    NO

    Context:
    {context}

    Question:
    {query.question}
    """

    verification = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": verification_prompt
            }
        ]
    )

    decision = (
        verification["message"]["content"]
        .strip()
        .upper()
    )

    print("VERIFICATION:", decision)

    if decision != "YES":

        def no_answer():
            yield (
                "I could not find relevant "
                "information in the provided documents."
            )

        return StreamingResponse(
            no_answer(),
            media_type="text/plain"
        )

    # Prompt

    
    prompt = f"""
You are a strict RAG assistant.

Answer ONLY using the provided context.

If the answer is not explicitly present
in the context, respond ONLY with:

"I could not find relevant information
in the provided documents."

Do not use external knowledge.
Do not make assumptions.
Do not generate information outside context.

context:
{context}

Conversation History:
{query.history}

Current Question:
{query.question}
"""

    # Ask Llama
    def generate():

        stream = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
            options={
                "num_predict": 500,
                "temperature": 0.1
            }
        )

        for chunk in stream:

            content = chunk["message"]["content"]

            if content:
                yield content

    # -----------------------------
    # RETURN STREAM
    # -----------------------------
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
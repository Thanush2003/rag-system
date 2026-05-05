from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from pymilvus import connections, Collection

from sentence_transformers import SentenceTransformer
import ollama

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

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Request schema
class Query(BaseModel):
    question: str
    history: str = ""

@app.post("/chat")
def chat(query: Query):

    # Convert question to embedding
    query_embedding = embedding_model.encode(
        [query.question]
    ).tolist()

    # Search Milvus
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=3,
        output_fields=["text", "source", "page"]
    )
    top_score = results[0][0].distance

# Threshold check
    if top_score < 0.5:

        def no_answer():
            yield (
                "I could not find relevant "
                "information in the provided documents."
            )

        return StreamingResponse(
            no_answer(),
            media_type="text/plain"
        )

    context = ""

    for hit in results[0]:
        context += (hit.entity.get("text") + "\n\n")

    # Prompt
    prompt = f"""
You are a helpful AI assistant.

Use previous conversation only if the current question depends on it.

If the question is unrelated to the previous conversation, answer according to the retrieved context only but dont mention "according to the context" phrase start directly giving the answer.

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
                "num_predict": 300,
                "temperature": 0.2
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
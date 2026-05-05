# DESIGN DECISIONS

## Chunk Size: 500 with 50 Overlap
A chunk size of 500 characters with 50-character overlap was selected to preserve contextual continuity while maintaining retrieval precision. Smaller chunks reduced semantic completeness, while larger chunks introduced retrieval noise.

---

## Embedding Model: all-MiniLM-L6-v2
The all-MiniLM-L6-v2 embedding model was chosen because it provides strong semantic retrieval quality with low latency and low memory requirements suitable for local deployment. Larger embedding models were considered but required more computational resources.

---

## Vector Database: Milvus
Milvus was selected because it supports scalable vector similarity search, metadata storage, and efficient indexing for semantic retrieval systems. Simpler alternatives such as FAISS lacked integrated metadata management capabilities.

---

## Similarity Metric: COSINE
Cosine similarity was chosen because semantic embeddings perform better when compared using angular similarity rather than Euclidean distance. L2 distance was tested initially but produced weaker threshold behavior.

---

## Index Type: IVF_FLAT
IVF_FLAT indexing was selected because it provides a good balance between retrieval accuracy and search speed for medium-scale vector collections.

---

## Top-k Retrieval: 3
Top-k retrieval was set to 3 in order to provide sufficient contextual diversity while minimizing irrelevant context injection into the prompt. Lower values reduced multi-chunk reasoning capability, while larger values increased noise.

---

## Prompt Template
A constrained prompt template was designed to encourage the model to answer strictly from retrieved context and reduce hallucinations. Simpler prompts produced less grounded responses.

---

## Generation Model: Llama3
Llama3 was selected because it provides strong instruction-following capability and high-quality local inference performance. Smaller models such as Phi3-mini were evaluated as lightweight alternatives.

---

## Generation Parameters
Temperature was set to 0.2 and max generation length to 300 tokens in order to produce concise and deterministic responses suitable for factual QA tasks. Higher temperatures increased hallucination risk.

---

## Metadata Storage
Source filenames and page numbers were stored in Milvus metadata fields to support retrieval tracing, evaluation, and Recall@3 computation.

---

## Streaming Responses
FastAPI StreamingResponse was used to stream generated tokens incrementally to the frontend in order to reduce perceived latency and improve user experience. Non-streaming generation introduced unnecessary waiting time for long answers.

---

## Out-of-Corpus Filtering
A cosine similarity threshold was introduced to reject irrelevant queries and reduce hallucinated answers for questions outside the document corpus. Without thresholding, the model attempted unsupported answers more frequently.
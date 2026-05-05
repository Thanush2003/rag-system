import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# -----------------------------
# CONNECT TO MILVUS
# -----------------------------
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD ALL PDFs FROM DATA FOLDER
# -----------------------------
documents = []

data_folder = "../data"

for file in os.listdir(data_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(
            data_folder,
            file
        )

        print(f"Loading: {file}")

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        # Add filename metadata
        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)

print(f"\nTotal pages loaded: {len(documents)}")

# -----------------------------
# SPLIT INTO CHUNKS
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print(f"Total chunks created: {len(docs)}")

# -----------------------------
# PREPARE TEXTS
# -----------------------------
texts = [
    doc.page_content
    for doc in docs
]

sources = [
    doc.metadata["source"]
    for doc in docs
]

pages = [
    doc.metadata["page"]
    for doc in docs
]

# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
).tolist()

print("Embeddings generated!")

# -----------------------------
# COLLECTION CONFIG
# -----------------------------
collection_name = "pdf_rag"

# Delete old collection if exists
if utility.has_collection(collection_name):

    utility.drop_collection(
        collection_name
    )

    print("Old collection deleted.")

# -----------------------------
# CREATE SCHEMA
# -----------------------------
fields = [

    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),

    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=5000
    ),

    FieldSchema(
        name="source",
        dtype=DataType.VARCHAR,
        max_length=500
    ),

    FieldSchema(
        name="page",
        dtype=DataType.INT64
    ),

    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=384
    )
]

schema = CollectionSchema(
    fields=fields,
    description="PDF RAG Collection"
)

# -----------------------------
# CREATE COLLECTION
# -----------------------------
collection = Collection(
    name=collection_name,
    schema=schema
)

# -----------------------------
# INSERT DATA
# -----------------------------
print("\nInserting data into Milvus...")

data = [
    texts,
    sources,
    pages,
    embeddings
]

collection.insert(data)

print("Data inserted!")

# -----------------------------
# CREATE VECTOR INDEX
# -----------------------------
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 128
    }
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

# -----------------------------
# LOAD COLLECTION
# -----------------------------
collection.load()

print("\nMilvus indexing completed!")

print("\nRAG ingestion finished successfully!")
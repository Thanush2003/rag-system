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
    "BAAI/bge-base-en-v1.5"
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
# SPLIT INTO PARENT CHUNKS
# -----------------------------
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

parent_docs = parent_splitter.split_documents(documents)

print(f"Total parent chunks created: {len(parent_docs)}")

# -----------------------------
# SPLIT INTO CHILD CHUNKS
# -----------------------------
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

child_docs = []

for parent_id, parent_doc in enumerate(parent_docs):

    children = child_splitter.create_documents(
        [parent_doc.page_content],
        metadatas=[parent_doc.metadata]
    )

    for child in children:

        child.metadata["parent_id"] = parent_id

        child.metadata["parent_text"] = (
            parent_doc.page_content
        )

        child.metadata["source"] = (
            parent_doc.metadata["source"]
        )

        child.metadata["page"] = (
            parent_doc.metadata["page"]
        )

        child_docs.append(child)

print(
    f"Total child chunks: {len(child_docs)}"
)

# -----------------------------
# PREPARE TEXTS
# -----------------------------
texts = []

parent_texts = []

sources = []

pages = []

parent_ids = []

for doc in child_docs:

    texts.append(
        doc.page_content
    )

    parent_texts.append(
        doc.metadata["parent_text"]
    )

    sources.append(
        doc.metadata["source"]
    )

    page = doc.metadata.get(
        "page",
        0
    )

    pages.append(
        int(page)
    )

    parent_ids.append(
        int(doc.metadata["parent_id"])
    )

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
        name="parent_text",
        dtype=DataType.VARCHAR,
        max_length=10000
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
        name="parent_id",
        dtype=DataType.INT64
    ),

    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=768
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
    parent_texts,
    sources,
    pages,
    parent_ids,
    embeddings
]

batch_size = 1000

total = len(texts)

for start in range(0, total, batch_size):

    end = start + batch_size

    batch_data = [

        texts[start:end],

        parent_texts[start:end],

        sources[start:end],

        pages[start:end],

        parent_ids[start:end],

        embeddings[start:end]
    ]

    collection.insert(batch_data)

    print(
        f"Inserted {end}/{total}"
    )

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
# Script to CREATE the 'face_embeddings' collection (e.g., create_milvus_collection.py)
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

MILVUS_HOST = "localhost" # Or "milvus-standalone" if running inside another container
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_embeddings"
EMBEDDING_DIM = 128 # Must match your dlib embedding dimension

def create_face_embeddings_collection():
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

        if utility.has_collection(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return Collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]

        schema = CollectionSchema(fields, description="Face embeddings collection for recognition")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # Create an index for faster search
        index_params = {
            "metric_type": "L2", # Or "COSINE" - must match your recognition metric
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Collection '{COLLECTION_NAME}' created successfully with index.")
        return collection

    except Exception as e:
        print(f"Error creating collection: {e}")
        return None

if __name__ == "__main__":
    create_face_embeddings_collection()
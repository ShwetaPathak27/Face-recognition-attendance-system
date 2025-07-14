from pymilvus import MilvusClient

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_embeddings"

def clear_milvus_collection():
    try:
        client = MilvusClient(uri=f"tcp://{MILVUS_HOST}:{MILVUS_PORT}")
        print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

        if client.has_collection(collection_name=COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' exists. Attempting to drop it...")
            client.drop_collection(collection_name=COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' dropped successfully.")
            
            
            print(f"Please now run your 'create_milvus_collection.py' script to re-create the collection.")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist. Nothing to clear.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()
            print("✅ Milvus client closed.")

if __name__ == "__main__":
    clear_milvus_collection()

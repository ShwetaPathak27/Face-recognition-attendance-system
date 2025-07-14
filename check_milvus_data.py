import time
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_embeddings"

# --- Helper Functions ---

def connect_milvus_client(host, port):
    client = None
    try:
        client = MilvusClient(uri=f"tcp://{host}:{port}")
        print("✅ Connected to Milvus.")
        return client
    except Exception as e:
        print(f"❌ Error connecting to Milvus: {e}")
        print("Ensure Milvus is running and accessible at the specified host/port.")
        return None

def check_milvus_data():
    milvus_client = connect_milvus_client(MILVUS_HOST, MILVUS_PORT)
    if not milvus_client:
        print("Milvus connection failed. Exiting.")
        return

    try:
        if not milvus_client.has_collection(collection_name=COLLECTION_NAME):
            print(f"❌ Error: Collection '{COLLECTION_NAME}' does not exist.")
            print("Please run 'create_milvus_collection.py' script first.")
            milvus_client.close()
            return

        print(f"✅ Collection '{COLLECTION_NAME}' verified (exists).")

        # --- NEW: Load the collection into Milvus memory for querying ---
        print(f"🔄 Loading collection '{COLLECTION_NAME}' into Milvus memory...")
        milvus_client.load_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' loaded successfully.")
        time.sleep(2) 
        

        stats = milvus_client.get_collection_stats(collection_name=COLLECTION_NAME)
        num_entities = stats.get('row_count', 0)
        print(f"\n📊 Total entities (faces) in '{COLLECTION_NAME}': {num_entities}")

        if num_entities > 0:
            print("\n🔍 Fetching up to 5 sample entries (user_id):")
            query_results = milvus_client.query(
                collection_name=COLLECTION_NAME,
                filter="",
                output_fields=["user_id"],
                limit=5
            )

            if query_results:
                for i, entity in enumerate(query_results):
                    print(f"  {i+1}. User ID: {entity.get('user_id', 'N/A')}")
            else:
                print("  No sample entries found, even though entity count is positive. This might indicate an issue with query or data structure.")
        else:
            print("No entities found in the collection. Please enroll faces using 'enroll_face.py'.")

    except Exception as e:
        print(f"❌ An error occurred while checking Milvus data: {e}")
    finally:
        
        if milvus_client and milvus_client.has_collection(collection_name=COLLECTION_NAME): 
             print(f"♻️ Releasing collection '{COLLECTION_NAME}' from Milvus memory.")
             milvus_client.release_collection(collection_name=COLLECTION_NAME)
        
        if milvus_client:
            milvus_client.close()
            print("Milvus client closed.")

if __name__ == "__main__":
    check_milvus_data()

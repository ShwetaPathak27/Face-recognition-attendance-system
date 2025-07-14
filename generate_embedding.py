import os
import cv2
import numpy as np
import dlib
from pymilvus import connections, Collection, utility, MilvusException
from pymilvus.exceptions import MilvusException

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "face_embeddings"
EMBEDDING_DIM = 128
FACE_DATA_DIR = "face_data"
MODEL_DIR = "model"

# Load Dlib models
cnn_face_detector = dlib.cnn_face_detection_model_v1(os.path.join(MODEL_DIR, "mmod_human_face_detector.dat"))
shape_predictor = dlib.shape_predictor(os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat"))
face_rec_model = dlib.face_recognition_model_v1(os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat"))

def get_milvus_collection():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        if not utility.has_collection(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' does not exist.")
            return None

        collection = Collection(name=COLLECTION_NAME)
        collection.load()
        print(f"Connected to Milvus collection: {COLLECTION_NAME}")
        return collection

    except MilvusException as e:
        print(f"Milvus error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def get_face_embedding_dlib(image_path):
    try:
        print(f"   Reading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"   Could not read image: {image_path}")
            return None

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = cnn_face_detector(rgb_img, 1)
        print(f"   Detected {len(detections)} face(s)")

        if len(detections) == 0:
            return None

        rect = detections[0].rect
        shape = shape_predictor(rgb_img, rect)
        descriptor = face_rec_model.compute_face_descriptor(rgb_img, shape)
        return list(descriptor)

    except Exception as e:
        print(f"   Error processing {image_path}: {e}")
        return None

def main():
    collection = get_milvus_collection()
    if not collection:
        return

    print("\n--- Starting embedding generation ---")
    batch_data = []
    inserted_count = 0
    skipped_count = 0

    if not os.path.isdir(FACE_DATA_DIR):
        print(f"Error: '{FACE_DATA_DIR}' not found.")
        return

    for user_id in os.listdir(FACE_DATA_DIR):
        user_folder = os.path.join(FACE_DATA_DIR, user_id)
        if not os.path.isdir(user_folder):
            continue

        print(f"\nProcessing user: {user_id}")
        for filename in os.listdir(user_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(user_folder, filename)
                embedding = get_face_embedding_dlib(img_path)

                if embedding:
                    batch_data.append({"user_id": user_id, "embedding": embedding})
                    inserted_count += 1
                    print(f"   ✅ Embedded: {filename}")
                else:
                    skipped_count += 1
                    print(f"   ❌ Skipped: {filename}")

    if batch_data:
        print(f"\nInserting {inserted_count} embeddings to Milvus...")
        try:
            result = collection.insert(batch_data)
            collection.flush()
            print(f"✅ Done. Inserted IDs: {result.primary_keys}")
        except Exception as e:
            print(f"❌ Insert error: {e}")
    else:
        print("❌ No embeddings to insert.")

    print(f"\n--- Done ---")
    print(f"✅ Total: {inserted_count} inserted, ❌ {skipped_count} skipped")

if __name__ == "__main__":
    main()

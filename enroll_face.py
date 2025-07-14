import cv2
import dlib
import numpy as np
import time
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import os

# --- Configuration ---
MILVUS_HOST = "localhost" # Milvus is typically exposed on localhost for Windows host
MILVUS_PORT = "19530" # Your Milvus gRPC port
COLLECTION_NAME = "face_embeddings" # <-- IMPORTANT: This MUST match your Milvus collection name
DIMENSION = 128 # Must match the dimension used by your dlib face recognition model

# Paths to Dlib models on your Windows machine
# Make sure these model files (.dat) are in a 'model' folder next to this script
PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
FACE_RECOCOGNITION_MODEL_PATH = "model/dlib_face_recognition_resnet_model_v1.dat"

# --- Helper Functions ---

def load_dlib_model(predictor_path, face_rec_model_path):
    """
    Loads Dlib's face detector, shape predictor, and face recognition model.
    Returns detector, shape_predictor, face_recognizer objects.
    """
    if not os.path.exists(predictor_path):
        print(f"❌ Error: Dlib model '{predictor_path}' not found.")
        print("Please ensure the 'model' folder and its contents are next to this script.")
        return None, None, None
    if not os.path.exists(face_rec_model_path):
        print(f"❌ Error: Dlib model '{face_rec_model_path}' not found.")
        print("Please ensure the 'model' folder and its contents are next to this script.")
        return None, None, None
        
    try:
        detector = dlib.get_frontal_face_detector() # Using the standard HOG detector
        sp = dlib.shape_predictor(predictor_path)
        facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        print("✅ Dlib model loaded successfully.")
        return detector, sp, facerec
    except Exception as e:
        print(f"❌ Error loading Dlib models: {e}")
        print("Please ensure 'model/' directory exists next to this script and contains:")
        print(f"- {predictor_path.split('/')[-1]}")
        print(f"- {face_rec_model_path.split('/')[-1]}")
        return None, None, None

def connect_milvus_client(host, port):
    """
    Connects to Milvus and returns the client.
    """
    client = None
    try:
        # Using tcp:// for MilvusClient as per standard gRPC connection
        client = MilvusClient(uri=f"tcp://{host}:{port}") 
        print("✅ Connected to Milvus.")
        return client
    except Exception as e:
        print(f"❌ Error connecting to Milvus: {e}")
        print("Ensure Milvus is running and accessible at the specified host/port.")
        return None

def get_face_embedding(frame, face, shape_predictor, face_recognizer):
    """
    Extracts facial landmarks and generates a 128-D embedding for a given face.
    Returns the NumPy array embedding.
    """
    # Convert frame to RGB for dlib, if it's BGR from OpenCV
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    shape = shape_predictor(rgb_frame, face)
    face_descriptor = face_recognizer.compute_face_descriptor(rgb_frame, shape)
    return np.array(face_descriptor).astype(np.float32)

# --- Main Enrollment Logic ---
def enroll_new_face():
    """
    Captures a face from the webcam, prompts for a user_id,
    generates embedding, and inserts into Milvus.
    """
    # Load Dlib models
    detector, sp, facerec = load_dlib_model(PREDICTOR_PATH, FACE_RECOCOGNITION_MODEL_PATH)
    if not all([detector, sp, facerec]):
        print("Exiting due to Dlib model loading failure.")
        return

    # Connect to Milvus
    milvus_client = connect_milvus_client(MILVUS_HOST, MILVUS_PORT)
    if not milvus_client:
        print("Exiting due to Milvus connection failure.")
        return
    
    # --- UPDATED Milvus Collection and Schema Check ---
    try:
        # Check if collection exists
        if not milvus_client.has_collection(collection_name=COLLECTION_NAME):
            print(f"❌ Error: Collection '{COLLECTION_NAME}' does not exist.")
            print("Please run 'create_milvus_collection.py' script first to create it.")
            milvus_client.close() # Close client before exiting
            return

        # Simplified schema check for MilvusClient 2.x
        # We rely on has_collection and assume create_milvus_collection.py ensures correct schema.
        print(f"✅ Collection '{COLLECTION_NAME}' verified (exists).")

    except Exception as e:
        print(f"Error checking Milvus collection existence: {e}")
        milvus_client.close() # Close client before exiting
        return
    # --- END UPDATED Milvus Collection and Schema Check ---

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use cv2.CAP_DSHOW for better compatibility on Windows
    if not cap.isOpened():
        print("❌ Error: Could not open webcam on Windows host. Ensure no other app is using it.")
        milvus_client.close() # Close client before exiting
        return

    print("\n🚀 Live stream for enrollment. Press 's' to Save a face, 'q' to Quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Retrying...")
                time.sleep(0.1)
                continue

            display_frame = frame.copy() # Work on a copy for display

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1) # Detect faces

            if len(faces) == 1:
                # Draw green rectangle if exactly one face is detected
                face = faces[0]
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "Ready to save (Press 's')", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(faces) == 0:
                cv2.putText(display_frame, "No face detected", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Multiple faces detected", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # Draw yellow rectangles for all detected faces if multiple
                for face in faces:
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


            cv2.imshow("Enrollment Camera", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(faces) == 1:
                    # Capture and process the frame with exactly one face
                    face_to_enroll = faces[0]
                    
                    person_id = input("Enter person's User ID (e.g., John_Doe_ID, U001): ")
                    if not person_id:
                        print("User ID cannot be empty. Skipping enrollment.")
                        continue
                    
                    embedding = get_face_embedding(frame, face_to_enroll, sp, facerec)
                    
                    # Prepare data for Milvus insertion
                    entity = {
                        "user_id": person_id, # <-- IMPORTANT: Use 'user_id'
                        "embedding": embedding.tolist()
                    }
                    
                    try:
                        insert_result = milvus_client.insert(
                            collection_name=COLLECTION_NAME,
                            data=[entity]
                        )
                        
                        # --- MODIFICATION START ---
                        # This line was commented out in your previous code.
                        # It's crucial to call flush() to ensure data is written to disk and visible.
                        milvus_client.flush(collection_name=COLLECTION_NAME) 
                        print(f"✅ Data flushed for collection '{COLLECTION_NAME}'.")
                        # --- MODIFICATION END ---


                        # --- UPDATED LOGIC TO GET INSERTED ID ---
                        print(f"DEBUG: Type of insert_result: {type(insert_result)}")
                        print(f"DEBUG: Content of insert_result: {insert_result}")
                        
                        inserted_id = None
                        if hasattr(insert_result, 'ids') and isinstance(insert_result.ids, list) and insert_result.ids:
                            inserted_id = insert_result.ids[0]
                        elif isinstance(insert_result, dict) and 'insert_ids' in insert_result and insert_result['insert_ids']:
                            inserted_id = insert_result['insert_ids'][0]
                        elif isinstance(insert_result, dict) and 'ids' in insert_result and insert_result['ids']:
                            inserted_id = insert_result['ids'][0]
                        elif isinstance(insert_result, dict) and 'primary_keys' in insert_result and insert_result['primary_keys']:
                            inserted_id = insert_result['primary_keys'][0]
                        
                        if inserted_id is not None:
                            print(f"✅ Successfully enrolled '{person_id}'. Inserted ID: {inserted_id}")
                            time.sleep(2) # Give user time to see success message
                        else:
                            print(f"❌ Warning: Insert seemed successful but no primary ID was found in the result structure. Result: {insert_result}")
                            time.sleep(2) # Still give time to read warning
                        # --- END UPDATED LOGIC ---
                            
                    except Exception as milvus_e:
                        print(f"❌ Error inserting into Milvus: {milvus_e}")
                        print("Ensure 'user_id' field in Milvus matches exactly, including VARCHAR type and max_length.")
                        milvus_client.close() # Close client on insert error
                        break # Exit loop on critical error

                else:
                    print("Please ensure exactly one face is visible to save.")
                
            elif key == ord('q'):
                print("Quitting enrollment.")
                break

    except KeyboardInterrupt:
        print("Enrollment process stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if milvus_client:
            # It's good practice to release collection and close client when done
            try:
                milvus_client.release_collection(COLLECTION_NAME) # Release collection from memory
            except Exception as e:
                print(f"Warning: Could not release collection: {e}")
            milvus_client.close()
        print("Enrollment script finished.")

if __name__ == "__main__":
    enroll_new_face()
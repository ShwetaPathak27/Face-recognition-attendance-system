import cv2
import dlib
import socket
import struct
import pickle
import time
import numpy as np
import threading
from pymilvus import MilvusClient, DataType

# --- Configuration ---
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_embeddings"
DIMENSION = 128
MILVUS_METRIC_TYPE = "L2"
RECOGNITION_THRESHOLD = 0.5

PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "model/dlib_face_recognition_resnet_model_v1.dat"

DLIB_DOWNSCALE_FACTOR = 1.0  # Set to 1.0 for full resolution detection
SKIP_FRAMES = 5
frame_counter = 0

# Shared recognition result
recognized_id_or_name = "Detecting..."
display_color = (255, 255, 0)

def load_dlib_model(predictor_path, face_rec_model_path):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    print("✅ Dlib model loaded successfully.")
    return detector, sp, facerec

def connect_and_load_milvus_collection(host, port, collection_name):
    try:
        client = MilvusClient(uri=f"tcp://{host}:{port}")
        print("✅ Connected to Milvus.")
        if not client.has_collection(collection_name=collection_name):
            print(f"❌ Collection '{collection_name}' not found.")
            client.close()
            return None
        client.load_collection(collection_name=collection_name)
        print(f"✅ Collection '{collection_name}' loaded.")
        return client
    except Exception as e:
        print(f"❌ Milvus connection error: {e}")
        return None

def get_face_embedding(frame, face, shape_predictor, face_recognizer):
    shape = shape_predictor(frame, face)
    face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
    return np.array(face_descriptor).astype(np.float32)

def recognize_face_in_milvus(milvus_client, live_embedding):
    global recognized_id_or_name, display_color
    try:
        results = milvus_client.search(
            data=[live_embedding.tolist()],
            collection_name=COLLECTION_NAME,
            limit=1,
            output_fields=["user_id"],
            search_params={"nprobe": 10}
        )
        if results and results[0]:
            hit = results[0][0]
            distance = hit.get('distance')
            user_id = hit.get('entity', {}).get('user_id', 'Unknown')

            if (MILVUS_METRIC_TYPE == "L2" and distance < RECOGNITION_THRESHOLD) or \
               (MILVUS_METRIC_TYPE == "COSINE" and distance < RECOGNITION_THRESHOLD):
                recognized_id_or_name = f"{user_id} ({distance:.2f})"
                display_color = (0, 255, 0)
                print(f"DEBUG: Matched: {recognized_id_or_name}. Distance: {distance:.2f}, Threshold: {RECOGNITION_THRESHOLD:.2f}")
            else:
                recognized_id_or_name = f"Unknown ({distance:.2f})"
                display_color = (0, 0, 255)
                print(f"DEBUG: No strong match found. Distance: {distance:.2f}, Threshold: {RECOGNITION_THRESHOLD:.2f}. Label: {recognized_id_or_name}")
        else: # This block is executed when Milvus search returns no hits at all
            recognized_id_or_name = "No match found in DB" # More specific message
            display_color = (0, 0, 255)
            # print(f"DEBUG: Milvus search returned no hits. Displaying: {recognized_id_or_name}") # This can be uncommented if needed
    except Exception as e:
        print(f"❌ Milvus search error: {e}")
        recognized_id_or_name = "Error"
        display_color = (0, 0, 255)

    # --- REMOVED THIS LINE ---
    # print(f"DEBUG: Milvus search result: {recognized_id_or_name}")
    # --- END REMOVAL ---

def run_live_face_recognition(): # Renamed from live_face_recognition to match your current code
    global recognized_id_or_name, display_color, frame_counter

    detector, sp, facerec = load_dlib_model(PREDICTOR_PATH, FACE_RECOGNITION_MODEL_PATH)
    if not all([detector, sp, facerec]):
        return

    milvus_client = connect_and_load_milvus_collection(MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME)
    if not milvus_client:
        return

    STREAMER_HOST = 'host.docker.internal'
    STREAMER_PORT = 8001

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to streamer at {STREAMER_HOST}:{STREAMER_PORT}...")
    try:
        client_socket.connect((STREAMER_HOST, STREAMER_PORT))
        print("✅ Connected to webcam streamer.")
    except Exception as e:
        print(f"❌ Streamer connection failed: {e}")
        return

    data = b""
    payload_size = struct.calcsize("!L")
    scale_factor = 2  # Scale up the display window

    print("\n🚀 Starting live face recognition. Press 'q' to quit.")

    try:
        while True:
            start_time = time.time()

            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    print("Streamer disconnected.")
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("!L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            buffer_array = pickle.loads(frame_data)
            original_frame = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)

            # --- ADD THE FIRST DEBUG PRINT HERE ---
            if original_frame is None:
                print(f"DEBUG: Failed to decode frame from streamer. Skipping.")
                continue # This acts like 'if not ret:' for traditional webcam reads
            # --- END OF FIRST ADDITION ---

            frame_counter += 1
            if frame_counter % SKIP_FRAMES != 0:
                # Just display
                display_frame = cv2.resize(
                    original_frame,
                    (int(original_frame.shape[1] * scale_factor), int(original_frame.shape[0] * scale_factor)),
                    interpolation=cv2.INTER_LINEAR
                )
                cv2.putText(display_frame, recognized_id_or_name, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
                cv2.imshow("Live Face Recognition", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Dlib Downscale
            small_frame = cv2.resize(original_frame, (
                int(original_frame.shape[1] * DLIB_DOWNSCALE_FACTOR),
                int(original_frame.shape[0] * DLIB_DOWNSCALE_FACTOR)
            ), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            # --- YOU ALREADY HAVE THIS DEBUG PRINT HERE, WHICH IS GREAT! ---
            print(f"DEBUG: Detected {len(faces)} face(s)")
            # --- END OF SECOND ADDITION (already existed) ---

            for face in faces:
                x1 = int(face.left() / DLIB_DOWNSCALE_FACTOR)
                y1 = int(face.top() / DLIB_DOWNSCALE_FACTOR)
                x2 = int(face.right() / DLIB_DOWNSCALE_FACTOR)
                y2 = int(face.bottom() / DLIB_DOWNSCALE_FACTOR)

                # Draw bounding box and name
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), display_color, 2)
                cv2.putText(original_frame, recognized_id_or_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)

                # Generate embedding and recognize
                embedding = get_face_embedding(original_frame, dlib.rectangle(x1, y1, x2, y2), sp, facerec)
                threading.Thread(target=recognize_face_in_milvus, args=(milvus_client, embedding)).start()

            # Display scaled up
            display_frame = cv2.resize(
                original_frame,
                (int(original_frame.shape[1] * scale_factor), int(original_frame.shape[0] * scale_factor)),
                interpolation=cv2.INTER_LINEAR
            )

            fps = 1 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, display_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Live Face Recognition", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        if milvus_client:
            try:
                milvus_client.release_collection(COLLECTION_NAME)
            except:
                pass # Already released or not loaded
            milvus_client.close()
        print("✅ Resources cleaned. Goodbye.")

if __name__ == "__main__":
    run_live_face_recognition()
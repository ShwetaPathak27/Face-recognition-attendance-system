import cv2
import dlib
import socket
import struct
import pickle
import time
import numpy as np
import threading
from pymilvus import MilvusClient, DataType

# --- NEW IMPORTS FOR POSTGRESQL ---
import psycopg2
from datetime import datetime, timedelta
import os
# --- END NEW IMPORTS ---

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

# Shared recognition result for display
recognized_id_or_name = "Detecting..."
display_color = (255, 255, 0)

# --- NEW: PostgreSQL Configuration ---
# Ensure these match your PostgreSQL setup
DB_HOST = os.getenv("DB_HOST", "postgres_db") # Use service name if Postgres is in Docker Compose
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attendance_db") # <--- UPDATE THIS
DB_USER = os.getenv("DB_USER", "attendance_user")     # <--- UPDATE THIS
DB_PASSWORD = os.getenv("DB_PASSWORD", "shwetapathak") # <--- UPDATE THIS

COOLDOWN_MINUTES = 1 # Your 1-minute debouncing period
# --- END NEW: PostgreSQL Configuration ---


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

# --- NEW: PostgreSQL Connection Function ---
def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("✅ Connected to PostgreSQL database.")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return None
# --- END NEW: PostgreSQL Connection Function ---

def get_face_embedding(frame, face, shape_predictor, face_recognizer):
    shape = shape_predictor(frame, face)
    face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
    return np.array(face_descriptor).astype(np.float32)

# --- MODIFIED: recognize_face_in_milvus to accept db_connection ---
def recognize_face_in_milvus(milvus_client, live_embedding, db_connection):
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
                
                # --- NEW: Attendance Logging Logic ---
                if db_connection:
                    cursor = db_connection.cursor()
                    try:
                        # Check last entry for this user_id within cooldown period
                        cooldown_time = datetime.now() - timedelta(minutes=COOLDOWN_MINUTES)
                        cursor.execute(
                            "SELECT timestamp FROM attendance_logs WHERE user_id = %s AND timestamp > %s ORDER BY timestamp DESC LIMIT 1",
                            (user_id, cooldown_time)
                        )
                        last_entry = cursor.fetchone()

                        if not last_entry:
                            # No recent entry, so insert a new one
                            cursor.execute(
                                "INSERT INTO attendance_logs (user_id, status) VALUES (%s, %s)",
                                (user_id, 'IN') # You can change 'IN' to 'PRESENT' or add 'OUT' logic later
                            )
                            db_connection.commit()
                            print(f"✅ Attendance recorded for {user_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Update display for a few seconds to show "IN!"
                            recognized_id_or_name = f"{user_id} - IN!"
                            display_color = (0, 255, 0) # Green for IN
                        else:
                            print(f"ℹ️ {user_id} already recorded within last {COOLDOWN_MINUTES} minute(s). Skipping.")
                            # Update display to show they are present but not re-recorded
                            recognized_id_or_name = f"{user_id} - Present"
                            display_color = (0, 165, 255) # Orange for Present
                    except Exception as db_e:
                        db_connection.rollback() # Rollback in case of error
                        print(f"❌ Database operation error: {db_e}")
                        recognized_id_or_name = "DB Error!"
                        display_color = (0, 0, 255) # Red for errors
                    finally:
                        cursor.close()
                else: # If DB connection failed
                    recognized_id_or_name = f"{user_id} ({distance:.2f})" # Still show recognition
                    display_color = (0, 255, 0)
                    print(f"DEBUG: Matched: {recognized_id_or_name}. Distance: {distance:.2f}, Threshold: {RECOGNITION_THRESHOLD:.2f} (DB not connected)")
                # --- END NEW: Attendance Logging Logic ---

            else: # No strong match found (distance >= threshold)
                recognized_id_or_name = f"Unknown ({distance:.2f})"
                display_color = (0, 0, 255)
                print(f"DEBUG: No strong match found. Distance: {distance:.2f}, Threshold: {RECOGNITION_THRESHOLD:.2f}. Label: {recognized_id_or_name}")
        else: # Milvus search returned no hits at all
            recognized_id_or_name = "No match found in DB"
            display_color = (0, 0, 255)
            # print(f"DEBUG: Milvus search returned no hits. Displaying: {recognized_id_or_name}") # Uncomment if needed
    except Exception as e:
        print(f"❌ Milvus search error: {e}")
        recognized_id_or_name = "Error"
        display_color = (0, 0, 255)

# --- REMOVED THIS LINE from recognize_face_in_milvus ---
# print(f"DEBUG: Milvus search result: {recognized_id_or_name}")
# --- END REMOVAL ---

# --- MODIFIED: run_live_face_recognition to handle DB connection ---
def run_live_face_recognition():
    global recognized_id_or_name, display_color, frame_counter

    detector, sp, facerec = load_dlib_model(PREDICTOR_PATH, FACE_RECOGNITION_MODEL_PATH)
    if not all([detector, sp, facerec]):
        return

    milvus_client = connect_and_load_milvus_collection(MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME)
    if not milvus_client:
        return

    # --- NEW: Connect to PostgreSQL ---
    db_connection = connect_db()
    if not db_connection:
        print("❌ Failed to connect to database. Recognition will proceed without attendance logging.")
        # Decide if you want to exit here if DB connection is mandatory
        # return
    # --- END NEW ---

    STREAMER_HOST = 'host.docker.internal'
    STREAMER_PORT = 8001

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to streamer at {STREAMER_HOST}:{STREAMER_PORT}...")
    try:
        client_socket.connect((STREAMER_HOST, STREAMER_PORT))
        print("✅ Connected to webcam streamer.")
    except Exception as e:
        print(f"❌ Streamer connection failed: {e}")
        # --- NEW: Close DB connection if streamer fails ---
        if db_connection:
            db_connection.close()
        # --- END NEW ---
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
                    # --- NEW: Close DB connection if streamer disconnects ---
                    if db_connection:
                        db_connection.close()
                    # --- END NEW ---
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

            if original_frame is None:
                print(f"DEBUG: Failed to decode frame from streamer. Skipping.")
                continue

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
            print(f"DEBUG: Detected {len(faces)} face(s)")

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
                # --- MODIFIED: Pass db_connection to the thread ---
                threading.Thread(target=recognize_face_in_milvus, args=(milvus_client, embedding, db_connection)).start()
                # --- END MODIFIED ---

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
            except Exception as e:
                print(f"Warning: Could not release Milvus collection: {e}")
            milvus_client.close()
        # --- NEW: Close DB connection ---
        if db_connection:
            try:
                db_connection.close()
                print("✅ PostgreSQL connection closed.")
            except Exception as e:
                print(f"Warning: Could not close PostgreSQL connection: {e}")
        # --- END NEW ---
        print("✅ Resources cleaned. Goodbye.")

if __name__ == "__main__":
    run_live_face_recognition()

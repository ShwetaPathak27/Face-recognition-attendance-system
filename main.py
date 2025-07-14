import cv2
import dlib
import numpy as np
import time
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import os
import psycopg2
from datetime import datetime, timedelta
import base64 
import asyncio 
from concurrent.futures import ThreadPoolExecutor 

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "face_embeddings"
DIMENSION = 128
MILVUS_METRIC_TYPE = "L2"
RECOGNITION_THRESHOLD = 0.5 # Lower for L2 (closer to 0 for similar faces), higher for COSINE (closer to 1 for similar faces)

# Paths to Dlib models (ensure they are in the 'model' directory relative to this script)
PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "model/dlib_face_recognition_resnet_model_v1.dat"

# PostgreSQL Configuration
DB_HOST = os.getenv("DB_HOST", "postgres_db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attendance_db")
DB_USER = os.getenv("DB_USER", "attendance_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "shwetapathak")

COOLDOWN_MINUTES = 1 # Your 1-minute debouncing period for attendance logging (for both IN and OUT)

# Global variables for Dlib models, Milvus client, and PostgreSQL connection
detector = None
sp = None
facerec = None
milvus_client = None
db_connection_pool = None 
last_recognized_time = {} # {user_id: datetime_object} for cooldown (for both IN and OUT logs)

# Thread pool for CPU-bound tasks like face recognition to avoid blocking FastAPI's event loop
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

app = FastAPI()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Helper Functions ---

def load_dlib_models_global():
    """Loads Dlib models globally when the app starts."""
    global detector, sp, facerec
    if not os.path.exists(PREDICTOR_PATH):
        print(f"❌ Error: Dlib model '{PREDICTOR_PATH}' not found.")
        print("Please ensure the 'model' folder and its contents are next to this script.")
        return False
    if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
        print(f"❌ Error: Dlib model '{FACE_RECOGNITION_MODEL_PATH}' not found.")
        print("Please ensure the 'model' folder and its contents are next to this script.")
        return False
        
    try:
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(PREDICTOR_PATH)
        facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
        print("✅ Dlib models loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Error loading Dlib models: {e}")
        return False

def connect_milvus_client_global():
    """Connects to Milvus and ensures the collection exists."""
    global milvus_client
    try:
        milvus_client = MilvusClient(uri=f"tcp://{MILVUS_HOST}:{MILVUS_PORT}")
        print("✅ Connected to Milvus.")
        
        # Check and create collection if it doesn't exist
        if not milvus_client.has_collection(collection_name=COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' not found. Creating it...")
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=256), # Increased max_length
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
            ]
            schema = CollectionSchema(fields, description="Face embeddings for attendance system")
            milvus_client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                shards_num=2 
            )
            print(f"✅ Collection '{COLLECTION_NAME}' created.")
            
            # Create index after collection creation
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", 
                index_type="IVF_FLAT", 
                metric_type=MILVUS_METRIC_TYPE,
                params={"nlist": 128} 
            )
            milvus_client.create_index(
                collection_name=COLLECTION_NAME,
                index_params=index_params
            )
            print(f"✅ Index created for collection '{COLLECTION_NAME}'.")
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already exists.")
        
        milvus_client.load_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' loaded into memory.")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Milvus or setting up collection: {e}")
        return False

def connect_db_global():
    """Connects to PostgreSQL globally when the app starts."""
    global db_connection_pool
    try:
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True 
        db_connection_pool = conn
        print("✅ Connected to PostgreSQL database.")
        return True
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return False

def get_face_embedding(frame, face):
    """Extracts facial landmarks and generates a 128-D embedding."""
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    shape = sp(rgb_frame, face)
    face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
    return np.array(face_descriptor).astype(np.float32)

# --- FastAPI Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on app startup."""
    print("Starting up FastAPI application...")
    if not load_dlib_models_global():
        print("Failed to load Dlib models. Exiting.")
        exit(1) # Exit if critical models can't be loaded
    if not connect_milvus_client_global():
        print("Failed to connect to Milvus. Exiting.")
        exit(1)
    if not connect_db_global():
        print("Failed to connect to PostgreSQL. Attendance logging will be disabled.")
        
    
@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on app shutdown."""
    print("Shutting down FastAPI application...")
    if milvus_client:
        try:
            milvus_client.release_collection(COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' released.")
        except Exception as e:
            print(f"Warning: Could not release Milvus collection: {e}")
        milvus_client.close()
        print("✅ Milvus connection closed.")
    if db_connection_pool:
        try:
            db_connection_pool.close()
            print("✅ PostgreSQL connection closed.")
        except Exception as e:
            print(f"Warning: Could not close PostgreSQL connection: {e}")

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    return FileResponse("static/index.html")

class ProcessFrameRequest(BaseModel):
    image_data: str # Base64 encoded image frame

class ProcessFrameResponse(BaseModel):
    processed_image_data: str # Base64 encoded image with overlays
    attendance_log: Optional[Dict[str, Any]] = None 
    dashboard_summary: Dict[str, int] 

@app.post("/api/process_frame", response_model=ProcessFrameResponse)
async def process_frame_api(request: ProcessFrameRequest):
    """Receives a single image frame from the frontend, processes it, and returns results."""
    image_data_b64 = request.image_data.split(',')[1] 
    
    current_frame = None
    try:
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image data.")
        current_frame = frame.copy() 
        # Convert to RGB for Dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
       
        faces = await asyncio.to_thread(detector, rgb_frame, 0) # 0 for no upsampling

        display_text = "No face detected"
        display_color = (0, 0, 255) # Red (default)
        logged_attendance_info = None # To hold details if attendance was logged

        if len(faces) == 1:
            face = faces[0]
            embedding = await asyncio.to_thread(get_face_embedding, rgb_frame, face)
            
            # Search Milvus
            search_results = milvus_client.search(
                data=[embedding.tolist()],
                collection_name=COLLECTION_NAME,
                limit=1,
                output_fields=["user_id"],
                search_params={"nprobe": 10}
            )

            if search_results and search_results[0]:
                hit = search_results[0][0]
                distance = hit.get('distance')
                user_id = hit.get('entity', {}).get('user_id', 'Unknown')

                if (MILVUS_METRIC_TYPE == "L2" and distance < RECOGNITION_THRESHOLD) or \
                   (MILVUS_METRIC_TYPE == "COSINE" and distance > RECOGNITION_THRESHOLD):
                    
                    current_time = datetime.now()
                    
                    # ---  Check Cooldown ---
                    if user_id in last_recognized_time:
                        time_since_last_log = current_time - last_recognized_time[user_id]
                        if time_since_last_log < timedelta(minutes=COOLDOWN_MINUTES):
                            # User is in cooldown, retrieve and display their last status
                            if db_connection_pool:
                                try:
                                    cursor = db_connection_pool.cursor()
                                    today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                                    cursor.execute(
                                        "SELECT status FROM attendance_logs WHERE user_id = %s AND timestamp >= %s ORDER BY timestamp DESC LIMIT 1",
                                        (user_id, today_start)
                                    )
                                    last_status_row = cursor.fetchone()
                                    last_status = last_status_row[0] if last_status_row else "Unknown"
                                    cursor.close()
                                    display_text = f"{user_id} ({last_status} - Cooldown)"
                                    display_color = (0, 165, 255) 
                                except Exception as e:
                                    print(f"Error getting last status during cooldown: {e}")
                                    display_text = f"{user_id} (DB Error)"
                                    display_color = (0, 0, 255)
                            else:
                                display_text = f"{user_id} (Cooldown)"
                                display_color = (0, 165, 255) 
                            
                            
                            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                            cv2.rectangle(current_frame, (x1, y1), (x2, y2), display_color, 2)
                            cv2.putText(current_frame, display_text, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
                            
                            # Encode processed frame to JPEG and return early
                            ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if not ret:
                                raise ValueError("Failed to encode processed frame to JPEG.")
                            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            dashboard_summary_data = await get_dashboard_summary()
                            return ProcessFrameResponse(
                                processed_image_data=f"data:image/jpeg;base64,{processed_image_b64}",
                                attendance_log=None, # No new log in cooldown
                                dashboard_summary=dashboard_summary_data
                            )
                   

                    # ---  Determine New Status and Log (Only if NOT in Cooldown) ---
                    new_status = "IN" # Default for first entry or after OUT

                    if db_connection_pool:
                        try:
                            cursor = db_connection_pool.cursor()
                            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                            cursor.execute(
                                "SELECT status FROM attendance_logs WHERE user_id = %s AND timestamp >= %s ORDER BY timestamp DESC LIMIT 1",
                                (user_id, today_start)
                            )
                            last_entry_today = cursor.fetchone()
                            cursor.close()

                            if last_entry_today:
                                last_status_from_db = last_entry_today[0]
                                if last_status_from_db == "IN":
                                    new_status = "OUT"
                                else:
                                    new_status = "IN"
                            

                            # Log the entry to DB
                            try:
                                cursor = db_connection_pool.cursor()
                                cursor.execute(
                                    "INSERT INTO attendance_logs (user_id, status) VALUES (%s, %s)",
                                    (user_id, new_status)
                                )
                                print(f"✅ Attendance logged for {user_id} - {new_status} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                last_recognized_time[user_id] = current_time 

                                display_text = f"{user_id} - {new_status}!"
                                display_color = (0, 255, 0) if new_status == "IN" else (0, 0, 255) 
                                logged_attendance_info = {
                                    "user_id": user_id,
                                    "timestamp": current_time.isoformat(),
                                    "status": new_status
                                }
                            except Exception as e:
                                print(f"❌ Database logging error for {user_id} ({new_status}): {e}")
                                display_text = "DB Error!"
                                display_color = (0, 0, 255) 
                            finally:
                                if cursor: cursor.close()

                        except Exception as e:
                            print(f"Error checking DB status for {user_id}: {e}")
                            display_text = f"{user_id} (DB Error)"
                            display_color = (0, 0, 255) 
                    else: 
                        display_text = f"{user_id} (DB Not Ready)"
                        display_color = (0, 255, 255) 
                    
                else: 
                    display_text = f"Unknown ({distance:.2f})"
                    display_color = (0, 0, 255) 
            else: 
                display_text = "No match found in DB"
                display_color = (0, 0, 255) 
            
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(current_frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        elif len(faces) > 1:
            display_text = "Multiple faces detected"
            display_color = (0, 255, 255) 
            cv2.putText(current_frame, display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2) 
            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), display_color, 2)
        else:
            display_text = "No face detected"
            display_color = (0, 0, 255) 
            cv2.putText(current_frame, display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2) 

        # Encode processed frame to JPEG for sending back
        ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            raise ValueError("Failed to encode processed frame to JPEG.")
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        
        dashboard_summary_data = await get_dashboard_summary() # Call the async function

        return ProcessFrameResponse(
            processed_image_data=f"data:image/jpeg;base64,{processed_image_b64}",
            attendance_log=logged_attendance_info,
            dashboard_summary=dashboard_summary_data
        )
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(ve)}")
    except Exception as e:
        print(f"Error during frame processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during frame processing: {e}")

class EnrollRequest(BaseModel):
    name: str
    image_data: str # Base64 encoded image

@app.post("/api/enroll_face")
async def enroll_face_api(request: EnrollRequest):
    """Endpoint to enroll a new face."""
    user_id = request.name
    image_data_b64 = request.image_data.split(',')[1] 
    
    try:
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image data.")

        # Convert to RGB for Dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        faces = await asyncio.to_thread(detector, rgb_frame, 1) # Use upsampling for better detection during enrollment

        if len(faces) == 1:
            face_to_enroll = faces[0]
            embedding = await asyncio.to_thread(get_face_embedding, rgb_frame, face_to_enroll)
            
            # Prepare data for Milvus insertion
            entity = {
                "user_id": user_id,
                "embedding": embedding.tolist()
            }
            
            insert_result = milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=[entity]
            )
            
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
                return {"success": True, "message": f"Face for '{user_id}' enrolled successfully! ID: {inserted_id}"}
            else:
                raise HTTPException(status_code=500, detail="Milvus insertion failed to return an ID.")
        elif len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image. Please try again.")
        else:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please ensure only one face is visible.")
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during face enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during enrollment: {e}")

class AttendanceLog(BaseModel):
    id: int
    user_id: str
    timestamp: datetime
    status: str

@app.get("/api/attendance_logs", response_model=List[AttendanceLog])
async def get_attendance_logs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None 
):
    """Fetches attendance logs from PostgreSQL with optional filters and limit."""
    if not db_connection_pool:
        raise HTTPException(status_code=503, detail="Database connection not available.")

    query = "SELECT id, user_id, timestamp, status FROM attendance_logs WHERE 1=1"
    params = []

    if start_date:
        query += " AND timestamp >= %s"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= %s"
        params.append(end_date)
    if user_id:
        query += " AND user_id ILIKE %s" # Case-insensitive search
        params.append(f"%{user_id}%")
    if status and status.upper() != "ALL":
        query += " AND status = %s"
        params.append(status.upper())
    
    query += " ORDER BY timestamp DESC"

    if limit is not None and limit > 0:
        query += f" LIMIT {limit}" # Apply limit

    try:
        cursor = db_connection_pool.cursor()
        cursor.execute(query, params)
        logs = cursor.fetchall()
        cursor.close()
        
        # Convert fetched data to AttendanceLog Pydantic models
        return [AttendanceLog(id=row[0], user_id=row[1], timestamp=row[2], status=row[3]) for row in logs]
    except Exception as e:
        print(f"Error fetching attendance logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching attendance logs: {e}")

@app.get("/api/dashboard_summary")
async def get_dashboard_summary():
    """Provides summary statistics for the dashboard."""
    if not db_connection_pool:
        # Return default summary if DB not available, don't raise HTTPException for dashboard
        return {
            "present_count": 0,
            "total_entries_today": 0
        }
    
    summary = {
        "present_count": 0,
        "total_entries_today": 0
    }

    try:
        cursor = db_connection_pool.cursor()
        
        # Get total entries today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cursor.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE timestamp >= %s",
            (today_start,)
        )
        summary["total_entries_today"] = cursor.fetchone()[0]

        # Get currently present count (simple logic: last entry for user is 'IN')
        cursor.execute("""
            SELECT user_id, status
            FROM attendance_logs
            WHERE (user_id, timestamp) IN (
                SELECT user_id, MAX(timestamp)
                FROM attendance_logs
                WHERE timestamp >= %s
                GROUP BY user_id
            )
            AND status = 'IN'
        """, (today_start,))
        
        present_users = cursor.fetchall()
        summary["present_count"] = len(present_users)

        cursor.close()
        return summary
    except Exception as e:
        print(f"Error fetching dashboard summary: {e}")
        # Return default summary on error, don't raise HTTPException for dashboard
        return {
            "present_count": 0,
            "total_entries_today": 0
        }


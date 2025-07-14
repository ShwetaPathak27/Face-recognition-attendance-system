Face Recognition Attendance System
This project implements a real-time face recognition-based attendance system. It leverages computer vision techniques to identify individuals from a live camera feed and logs their attendance in a PostgreSQL database. Face embeddings are stored and managed using Milvus, a vector database, for efficient similarity search.

The system features a web-based frontend for user interaction, allowing for live recognition monitoring and a controlled process for enrolling new faces.

✨ Features
Real-time Face Recognition: Identify individuals from a live camera stream.

User-Controlled Face Enrollment: Securely enroll new employee faces with a confirmation step.

Multi-Camera Support: Select desired camera for both recognition and enrollment from available devices.

Attendance Logging: Record "IN" entries for recognized individuals in a PostgreSQL database.

Debouncing/Cooldown: Prevent multiple rapid attendance logs for the same person within a configurable period.

Dashboard View: Display real-time attendance summary (present employees, total entries) and recent activity logs.

Scalable Embedding Storage: Utilizes Milvus for efficient storage and similarity search of face embeddings.

Modular Architecture: Separate frontend (HTML/JS) and backend (Python) components.

Dockerized Services: Easy setup for Milvus and PostgreSQL using Docker Compose.

🚀 Technologies Used
Backend:

Python

OpenCV (cv2)

Dlib (for face detection, landmark prediction, and face embedding generation)

FastAPI ( for web server )

pymilvus (Milvus Python SDK)

psycopg2 (PostgreSQL adapter for Python)

Frontend:

HTML

CSS (Tailwind CSS)

JavaScript (WebRTC for camera access, Fetch API for backend communication)

Databases:

Milvus (Vector Database for face embeddings)

PostgreSQL (Relational Database for attendance logs and user metadata)

Containerization:

Docker

Docker Compose

📂 Project Structure
.
├── static/                 # Frontend web files
│   ├── index.html          # Main dashboard and UI
│   └── script.js           # Frontend JavaScript logic (camera, API calls, UI updates)
├── backend/                # Python backend application
│   ├── main.py             # Main Flask/FastAPI server logic, API endpoints
│   └── (other_backend_files.py) # e.g., db_utils.py, face_processing.py if separated
├── model/                  # Dlib pre-trained models
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
├── dlib-19.24.2-cp312-cp312-win_amd64.whl # Dlib wheel file (if Windows specific)
├── mmod_human_face_detector.dat # Dlib's CNN face detector model (if used)
├── .gitignore              # Files/directories to ignore in Git
├── README.md               # This file
├── docker-compose.yml      # Docker Compose configuration for services
├── requirements.txt        # Python dependencies
├── check_milvus_data.py    # Utility to check Milvus data (Optional)
├── clear_milvus_data.py    # Utility to clear Milvus data (Optional)
├── create_milvus_collection.py # Utility to create Milvus collection (Optional)
├── enroll_face.py          # Script for enrolling faces (or integrated into main.py)
├── generate_embedding.py   # Script for generating embeddings (or integrated)
├── live_recognition.py     # Script for live recognition (or integrated into main.py)
└── (other_utility_scripts.py)

🎥 Live Demo
https://www.linkedin.com/posts/shweta-pathak-09a023295_computervision-ai-python-activity-7350449648219697154-KTQ9?

⚙️ Setup Guide

Follow these steps to get the system up and running on your local machine.

Prerequisites
Git: For cloning the repository.

Docker & Docker Compose: Essential for running Milvus and PostgreSQL.

Install Docker Desktop

Python 3.8+: For the backend.

Download Python

pip: Python package installer (comes with Python).

1. Clone the Repository
git clone https://github.com/ShwetaPathak27/Face-recognition-attendance-system.git
cd Face-recognition-attendance-system

2. Download Dlib Models
The .dat files for Dlib models are crucial but might be large. If they are not present in the model/ directory after cloning, you'll need to download them manually:

shape_predictor_68_face_landmarks.dat:

Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Unzip the .bz2 file.

dlib_face_recognition_resnet_model_v1.dat:

Download from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Unzip the .bz2 file.

Place both unzipped .dat files into the model/ directory within your project.

3. Docker Setup (Milvus & PostgreSQL)
Navigate to the project root directory where docker-compose.yml is located.

# Start Milvus and PostgreSQL services
docker-compose up -d

This will pull the necessary Docker images and start the services in the background. It might take a few minutes for services to be fully ready.

Database Initialization (PostgreSQL)
You need to create the attendance_logs table in your PostgreSQL database.

Connect to PostgreSQL:
You can use a PostgreSQL client (like psql or DBeaver) or execute commands within the Docker container.
To connect via psql from your host (if you have it installed):

psql -h localhost -p 5432 -U attendance_user -d attendance_db

(Enter password ...... when prompted)

Create Table:
Once connected, run the following SQL command:

CREATE TABLE IF NOT EXISTS attendance_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL -- e.g., 'IN', 'OUT', 'PRESENT'
);

Milvus Collection Setup
You need to create the Milvus collection for face embeddings. You can use the create_milvus_collection.py script provided in your project.

python create_milvus_collection.py

This script should connect to Milvus and create the face_embeddings collection with the specified DIMENSION and MILVUS_METRIC_TYPE.

4. Python Backend Setup
Create a Virtual Environment:
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv venv

Activate the Virtual Environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Python Dependencies:

pip install -r requirements.txt

Note for Windows Dlib: If dlib installation fails on Windows, you might need to install it from a pre-compiled wheel file. You have dlib-19.24.2-cp312-cp312-win_amd64.whl in your project. If pip install dlib fails, try:

pip install dlib-19.24.2-cp312-cp312-win_amd64.whl

Ensure you have the correct Visual C++ build tools if you're compiling dlib from source, or use the wheel.

Set Environment Variables:
Your backend uses environment variables for database credentials. Create a .env file in your project root (it's in .gitignore, so it won't be committed) and add:

DB_HOST=postgres_db
DB_PORT=5432
DB_NAME=attendance_db
DB_USER=attendance_user
DB_PASSWORD= your_actual_strong_password_here

Make sure your backend code loads these environment variables (e.g., using python-dotenv).

🚀 Usage
1. Start the Backend Server
Navigate to your project's root directory and ensure your virtual environment is active.

python backend/main.py # Or whatever your main backend file is named

This will start your Flask/FastAPI server, which handles API requests for face processing and enrollment.

2. Access the Frontend
Open your web browser and navigate to the address where your frontend is served. If running locally without a separate web server, you might open static/index.html directly in your browser, but for full API functionality, you'll need a way for the frontend to communicate with your backend.


3. Enroll New Faces
In the "Enroll New Face" panel, enter the Employee Name.

Select the desired camera from the "Select Camera" dropdown in the "Main Camera Management" section. This camera will also be used for enrollment.

Click the "Start Capture" button. The camera feed will appear in the "Enrollment Preview" area.

Adjust your face in front of the camera.

Click the "Confirm Capture" button to take the photo. The live feed will stop, and the captured image will be displayed.

Click the "Enroll" button to send the captured face and name to the backend for embedding generation and storage in Milvus.

4. Start Live Recognition
In the "Main Camera Management" panel, select the desired camera from the "Select Camera" dropdown.

Click the "Start Main Camera" button. The live recognition feed will start in the right panel.

The system will continuously detect and recognize faces, updating the dashboard and recent activity logs.


🎉 Conclusion
This Face Recognition Attendance System provides a robust and user-friendly solution for automated attendance tracking. By integrating real-time computer vision with scalable database technologies like Milvus and PostgreSQL, it offers an efficient and accurate way to manage employee presence. We hope this project serves as a valuable resource for understanding and implementing modern face recognition applications.

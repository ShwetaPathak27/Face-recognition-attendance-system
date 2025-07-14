FROM python:3.11-slim-buster

# Install build dependencies for dlib
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install psycopg2-binary

# Explicitly copy only the files you know you need at /app
COPY generate_embedding.py .
COPY main.py .
COPY milvus_client.py . 
COPY .env . 
COPY live_recognition.py .

# Copy the static folder (if it exists)
COPY static/ ./static/



# Create the directory for storing face data
RUN mkdir -p face_data

# Create models directory inside container
RUN mkdir -p model



# Copy models into that folder
COPY mmod_human_face_detector.dat model/
COPY shape_predictor_68_face_landmarks.dat model/
COPY dlib_face_recognition_resnet_model_v1.dat model/




# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run your FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
import faiss
import numpy as np
import csv
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
from io import BytesIO
from PIL import Image
from pandas
import numptyy

app = FastAPI()

# Function to create the embedding for a given image
def create_face_embedding(image, detector_backend='mtcnn'):
    try:
        img = np.array(image)
        embedding = DeepFace.represent(img_path=img, model_name='Facenet512', detector_backend=detector_backend)
        return embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

# Function to mark attendance in a CSV file
def mark_attendance(student_id, match_found):
    file_path = 'attendance.csv'

    # Get the current time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Open the CSV file and append the new attendance entry
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if match_found:
            writer.writerow([student_id, "Present", timestamp])
        else:
            writer.writerow([student_id, "No match", timestamp])

# Load the FAISS index
embedding_dim = 512  # Dimension of your embeddings (e.g., 512 for FaceNet512)
index_path = 'C:\\Users\\ergou\\PycharmProjects\\pythonProject\\live_attendance\\data\\faiss_index.index'

try:
    faiss_index = faiss.read_index(index_path)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")

# API endpoint for checking attendance
@app.post("/check_attendance/")
async def check_attendance(file: UploadFile = File(...)):
    try:
        # Load the image
        image = Image.open(BytesIO(await file.read()))

        # Step 1: Create the face embedding for the new image
        new_embedding_data = create_face_embedding(image, detector_backend='retinaface')

        # Extract the embedding vector from the result (assuming it's a list of dictionaries)
        if new_embedding_data:
            new_embedding = np.array(new_embedding_data[0]['embedding'], dtype=np.float32)

            # Step 2: Search the FAISS index for the nearest neighbors
            # Reshape the embedding to match the input format expected by FAISS
            new_embedding = new_embedding.reshape(1, -1)

            # Perform search on the index
            distances, similar_students = faiss_index.search(new_embedding, 5)  # Get top 5 nearest neighbors
            print("distance:", distances)
            print("embed: ", new_embedding)

            # Define a distance threshold (tune this value based on your data)
            distance_threshold = 60  # You can experiment with different thresholds

            # Step 3: Check if any of the nearest neighbors are valid matches
            match_found = False
            student_id = None

            for i, idx in enumerate(similar_students[0]):
                # Cast the FAISS index result to standard Python `int`
                if distances[0][i] < distance_threshold:
                    student_id = int(idx)  # Ensure it's an int, not a numpy type
                    mark_attendance(student_id=student_id, match_found=True)
                    match_found = True
                    return {"status": "success", "student_id": student_id, "distance": float(distances[0][i])}  # Ensure float

            if not match_found:
                # If no match found within threshold, mark attendance as "No match"
                mark_attendance(student_id="Unknown", match_found=False)
                return {"status": "failure", "message": "No valid match found"}
        else:
            return {"status": "failure", "message": "Failed to create embedding for the image."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Example command to run the API using Uvicorn:
# uvicorn main:app --reload

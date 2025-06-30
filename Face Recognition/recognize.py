import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.preprocessing import normalize


with open("embeddings.pkl", "rb") as f:
    database = pickle.load(f)

# def preprocess_face(face):
#     if face.size == 0:
#         return None
    
#     face = cv.resize(face, (160, 160))

#     face = face.astype(np.float32)
#     face = (face - 127.5) / 128.0

#     return face

def find_match(embeddings):
    min_dist = float('inf')
    identify = "Unknown"

    for name, db_embed in database.items():
        # Cosine distance
        dist = np.linalg.norm(embeddings - db_embed)

        # Cosine similarity
        cosine_sim = np.dot(embeddings, db_embed) / (np.linalg.norm(embeddings) * np.linalg.norm(db_embed))
        cosine_dist = 1 - cosine_sim  # Convert similarity to distance

        print(f"Distance to {name}: Euclidean={dist:.4f}, Cosine={cosine_dist:.4f}")

        if dist < 1.2 and dist < min_dist:
            min_dist = dist
            identify = name

    print(f"Best match: {identify} with distance {min_dist:.4f}")
    print("Live embedding sample:", embeddings[:5])
    print("========================\n")
    return identify

# Debug function to check your database
def debug_database():
    print("=== DATABASE DEBUG ===")
    for name, embedding in database.items():
        print(f"{name}:")
        print(f"  Shape: {embedding.shape}")
        print(f"  Mean: {np.mean(embedding):.6f}")
        print(f"  Std: {np.std(embedding):.6f}")
        print(f"  Norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Min/Max: [{np.min(embedding):.3f}, {np.max(embedding):.3f}]")
        print("---")

# Run debug
debug_database()

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)

cap = cv.VideoCapture(0)

# frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # frame_count += 1
    # Process every 10th frame to reduce processing load
    # if frame_count % 10 != 0:
    #     cv.imshow("Face Recognition", frame)
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     continue

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceDetection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            padding = 20
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)

            face = frame[y1_pad: y2_pad, x1_pad:x2_pad]
            if face.size == 0:
                continue

            # face_processed = preprocess_face(face)
            # if face_processed is None:
            #     continue
                
            # Get embedding
            try:
                # face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                embedding_obj = DeepFace.represent(
                    img_path=face,
                    model_name='Facenet',
                    enforce_detection=False,
                    detector_backend='skip',
                )
                embedding = np.array(embedding_obj[0]['embedding'])
                embedding = normalize([embedding])[0]

                print(f"Processed embedding shape: {embedding.shape}")
                print(f"Processed embedding norm: {np.linalg.norm(embedding):.6f}")

                name = find_match(embedding)
                
                # Draw bounding box and name
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, name, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                import traceback
                traceback.print_exc()
                continue


    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv.waitKey(0)
cv.destroyAllWindows()
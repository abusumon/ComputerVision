import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import normalize
import pickle


# def preprocess_face(face):
#     if face.size == 0:
#         return None
#     face = cv.resize(face, (160, 160))
#     face = face.astype('float32')
#     face = (face - 127.5) / 128.0
#     return face

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)

databases = {}
IMG_DIR = r"D:/Sumon/Coding/Computer Vision/Face Recognition/Face_Data"

for person in os.listdir(IMG_DIR):
    path = os.path.join(IMG_DIR, person)

    if not os.path.isdir(path):
        continue

    embeddings = []

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv.imread(img_path)

        if img is None:
            continue

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = faceDetection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x1 = max(0, int(bboxC.xmin * w))
                y1 = max(0, int(bboxC.ymin * h))
                x2 = min(w, int((bboxC.xmin + bboxC.width) * w))
                y2 = min(h, int((bboxC.ymin + bboxC.height) * h))

                face = img[y1:y2, x1:x2]

                try:
                    embedding_obj = DeepFace.represent(
                        img_path=face,
                        model_name='Facenet',
                        enforce_detection=False,
                    )
                    embedding = np.array(embedding_obj[0]['embedding'])
                    embedding = normalize([embedding])[0]
                    embeddings.append(embedding)
                    print(f"Processed {img_name} for {person}, embedding sample:", embedding[:5])
                except Exception as e:
                    print(f"Error processing {img_name} for {person}: {e}")
                    continue

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = normalize([avg_embedding])[0]
        databases[person] = avg_embedding
        print(f"Processed {len(embeddings)} faces for {person}")


with open("embeddings.pkl", "wb") as f:
    pickle.dump(databases, f)

print(f"Database created with {len(databases)} entries.")
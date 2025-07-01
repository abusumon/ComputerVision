import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import normalize
import pickle

class FaceRecognition:
    def __init__(self, database=None, img_dir=None, embeddings=None):
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)
        self.database = database or {}
        self.img_dir = img_dir 
        self.embeddings = embeddings or []
    
    def make_database(self):
        self.database = {}
        for person in os.listdir(self.img_dir):
            path = os.path.join(self.img_dir, person)

            if not os.path.isdir(path):
                continue

            self.embeddings = []

            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv.imread(img_path)

                if img is None:
                    continue

                self.img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                self.results = self.faceDetection.process(self.img_rgb)
                if self.results.detections:
                    for detection in self.results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = img.shape
                        x1 = max(0, int(bboxC.xmin * w))
                        y1 = max(0, int(bboxC.ymin * h))
                        x2 = min(w, int((bboxC.xmin + bboxC.width) * w))
                        y2 = min(h, int((bboxC.ymin + bboxC.height) * h))

                        face = img[y1:y2, x1:x2]

                        try:
                            embedding_obj = DeepFace.represent(
                                img_path=None,
                                img=face,
                                model_name='Facenet',
                                enforce_detection=False,
                                detector_backend='skip',
                            )
                            embedding = np.array(embedding_obj[0]['embedding'])
                            embedding = normalize([embedding])[0]
                            self.embeddings.append(embedding)
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
            if self.embeddings:
                avg_embedding = np.mean(self.embeddings, axis=0)
                avg_embedding = normalize([avg_embedding])[0]
                self.database[person] = avg_embedding
        
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(self.database, f)
        
        return self.database
    def load_database(self, path="embeddings.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.database = pickle.load(f)

    def find_match(self, input_embedding):
        min_dist = float('inf')
        identify = "Unknown"

        for name, db_embed in self.database.items():
            dist = np.linalg.norm(input_embedding - db_embed)

            if dist < 1.2 and dist < min_dist:
                min_dist = dist
                identify = name
        
        return identify
    
    def FaceRecog(self):
        cap = cv.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.faceDetection.process(img_rgb)

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

                    # Get embedding
                    try:
                        # face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                        embedding_obj = DeepFace.represent(
                            img=face,
                            model_name='Facenet',
                            enforce_detection=False,
                            detector_backend='skip',
                        )
                        embedding = np.array(embedding_obj[0]['embedding'])
                        embedding = normalize([embedding])[0]

                        print(f"Processed embedding shape: {embedding.shape}")
                        print(f"Processed embedding norm: {np.linalg.norm(embedding):.6f}")

                        name = self.find_match(embedding)
                        
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
        cv.destroyAllWindows()

face_rec = FaceRecognition(img_dir=r"D:/Sumon/Coding/Computer Vision/Face Recognition/Face_Data")
face_rec.load_database()
face_rec.FaceRecog()
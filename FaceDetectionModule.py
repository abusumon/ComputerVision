import cv2 as cv
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, confidence):
        self.confidence = confidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.confidence)
        self.mpDraw = mp.solutions.drawing_utils


    def findFace(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        bboxes = []

        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append(bbox)
                if draw:
                    cv.rectangle(img, bbox, (255, 0, 255), 2)
        return img, bboxes

def main():
    pTime = 0
    cap = cv.VideoCapture(0)
    detector = FaceDetector(confidence=0.75)
    

    while True:
        success, img = cap.read()
        img, bboxes = detector.findFace(img)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv.putText(img, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
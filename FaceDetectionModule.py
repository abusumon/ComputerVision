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
                bboxes.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox)
                    cv.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        return img, bboxes
    
    def fancyDraw(self, img, bbox, l=30, t=10, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv.rectangle(img, bbox, (255, 0, 255), rt)

        # Top left
        cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(img, (x, y), (x , y + l), (255, 0, 255), t)

        # Top right
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1 , y + l), (255, 0, 255), t)

        # Bottom left
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x , y1 - l), (255, 0, 255), t)

        # Bottom right
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1 , y1 - l), (255, 0, 255), t)

        return img

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
import cv2 as cv
import mediapipe as mp
import time

class FaceMeshModule:
    def __init__(self,staticMode=False, max_faces=2, minDetectionCon=0.5, minTrackingCon=0.5):
        self.max_faces = max_faces
        self.staticMode = staticMode
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.max_faces, self.minDetectionCon, self.minTrackingCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                        self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv.putText(img, str(id), (x, y),cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)

        return img, faces

def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceMeshModule()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=True)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f"FPS: {int(fps)}", (20, 70),cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
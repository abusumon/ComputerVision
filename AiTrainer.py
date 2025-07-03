import cv2 as cv
import numpy as np
import time
import PoseModule as pm


cap = cv.VideoCapture("PoseVideos/curls.mp4")


detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv.resize(img, (720, 480))

    # img = cv.imread("PoseVideos/test.jpg")
    img = detector.findPose(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))

        color = (255, 0, 255)
        if per == 100:
            color = (255, 0, 255)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (255, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0
        cv.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv.rectangle(img, (1100, int(bar)), (1175, 650), color, cv.FILLED)
        cv.putText(img, f'{int(per)}%', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        cv.rectangle(img, (50, 650), (85, 100), (0, 0, 255), cv.FILLED)
        cv.putText(img, str(count), (45, 67), cv.FONT_HERSHEY_PLAIN, 10, color, 25)
    
    cTime = time.time()
    fps = 1 /(cTime-pTime)
    pTime = cTime
    
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv.destroyAllWindows()
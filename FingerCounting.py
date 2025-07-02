import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm
import pyttsx3
import threading

################################################
wCam, hCam = 640, 480
################################################

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate

tipIds = [4, 8, 12, 16, 20]

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
# totalFingers = 0

prevFingers = -1
prevHandType = None
if prevHandType is None:
    speak("No hand detected")
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    handType = "Right" if detector.leftORright(img) else "Left"
    if handType != prevHandType:
        speak(f"{handType} Hand detected")
        prevHandType = handType

    if len(lmList) != 0:
        fingers = []

        if handType == "Left":
            # Thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # Remaining Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)

        img[0:250, 0:200] = cv.resize(overlayList[totalFingers-1], (200, 250), interpolation=cv.INTER_AREA)
        if totalFingers != prevFingers:
            speak(str(totalFingers))
            prevFingers = totalFingers

        # cv.rectangle(img, (20, 325), (170, 425), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(totalFingers), (45, 400), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv.imshow("Image", img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import autopy

####################################
wCam, hCam = 640, 480  # Width and height of the camera frame
frameR = 100  # Frame Reduction
smoothening = 7  # Smoothing factor for mouse movement
####################################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Variables
pTime = 0
plocX, plocY = 0, 0  # Previous location of the mouse
clocX, clocY = 0, 0  # Current location of the mouse

# Create a hand detector object
detector = htm.handDetector(maxHands=1)

# Get screen size
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find the hand landmarks
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break
        
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # 2. Get the tip of the index finger and the middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        
        # Draw rectangle for the region of interest
        cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # 4. Only Index Finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            
            # 7. Move the mouse
            try:
                autopy.mouse.move(clocX, clocY)
                cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                plocX, plocY = clocX, clocY
            except Exception as e:
                print(f"Error moving mouse: {e}")
        
        # 8. Both Index and Middle Finger: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find the distance between the index and middle finger
            length, img, lineInfo = detector.findDistance(8, 12, img)
            
            # 10. If the distance is short, click the mouse
            if length < 40:
                cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                try:
                    autopy.mouse.click()
                    time.sleep(0.1)  # Add small delay to prevent multiple clicks
                except Exception as e:
                    print(f"Error clicking mouse: {e}")

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
    # Display
    cv.imshow("Virtual Mouse", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
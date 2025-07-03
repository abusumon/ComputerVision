import mediapipe as mp
import cv2
import time
import math
import os

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        
        # FIXED: Updated for newer MediaPipe versions
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,  # 0, 1, or 2
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
        # Initialize results
        self.results = None
        self.lmList = []
    
    def findPose(self, img, draw=True):
        if img is None:
            return None
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if img is None:
            return self.lmList
            
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # FIXED: Add validation
        if len(self.lmList) <= max(p1, p2, p3):
            print(f"Error: Landmark indices out of range. Available: {len(self.lmList)}")
            return 0
            
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
            
        # Draw
        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    # FIXED: Added error handling
    video_path = 'PoseVideos/1.mp4'
    
    # Try webcam if video file doesn't exist
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found. Using webcam...")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    pTime = 0
    detector = poseDetector()
    
    while True:
        success, img = cap.read()
        
        # FIXED: Handle end of video or failed frame
        if not success:
            print("End of video or failed to read frame")
            break
            
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # FIXED: Add bounds checking
            if len(lmList) > 14:
                print(f"Right wrist landmark: {lmList[14]}")
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
            else:
                print(f"Warning: Only {len(lmList)} landmarks detected")
        
        # Calculate and display FPS
        cTime = time.time()
        if pTime != 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        
        cv2.imshow("Image", img)
        
        # FIXED: Added proper exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # FIXED: Proper cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
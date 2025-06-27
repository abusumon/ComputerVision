Computer Vision Projects: HandTracking & PoseEstimation
Welcome to my repository containing projects focused on real-time Hand Tracking and Pose Estimation using computer vision techniques and MediaPipe.

Features
Hand Tracking Module
Detect and track multiple hands in real-time, including landmark detection for finger joints.

Pose Estimation Module
Estimate human body pose landmarks for applications like activity recognition, gesture control, or fitness tracking.

Real-time Processing
Both modules are optimized for live webcam input with efficient performance.

Technologies Used

Python 3.x
OpenCV
MediaPipe
TensorFlow Lite (used internally by MediaPipe)

Usage
Clone the repository:

git clone https://github.com/abusumon/ComputerVision.git
Install dependencies:

pip install opencv-python mediapipe

Run HandTracking module:
python HandTrackingModule.py

Run PoseEstimation module:
python PoseEstimation.py

Project Structure

/ComputerVision
├── HandTrackingModule.py      # Hand tracking implementation using MediaPipe
├── PoseEstimation.py          # Pose estimation implementation
└── README.md                  # This file

Future Work

Add custom landmark detection for user-defined objects
Integrate gesture recognition models
Improve UI/UX for better visualization
Explore training custom pose models

Contact
Created by Sumon — feel free to reach out!
GitHub: https://github.com/abusumon


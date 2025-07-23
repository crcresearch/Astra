#!/usr/bin/env python3
"""
Debug script to test camera and emotion detection
"""

import cv2
import numpy as np
import sys

print("=== Camera Debug Test ===")
print(f"OpenCV version: {cv2.__version__}")
print(f"Python version: {sys.version}")

# Test 1: Basic camera
print("\n1. Testing basic camera access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera failed to open!")
    print("\nPossible solutions:")
    print("1. On macOS, go to System Preferences > Security & Privacy > Camera")
    print("   and make sure Terminal/Python has camera access")
    print("2. Try running: sudo python debug_camera.py")
    print("3. Check if another app is using the camera")
    
    # Try different camera indices
    print("\nTrying other camera indices...")
    for i in range(1, 5):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"✅ Camera found at index {i}")
            test_cap.release()
            break
        test_cap.release()
    exit(1)

print("✅ Camera opened successfully!")

# Test 2: Read a frame
print("\n2. Testing frame capture...")
ret, frame = cap.read()
if not ret:
    print("❌ Cannot read frame from camera!")
    cap.release()
    exit(1)

print("✅ Frame captured successfully!")
print(f"   Frame shape: {frame.shape}")

# Test 3: Display window
print("\n3. Testing display window...")
print("Press 'q' to continue to emotion detection test")
print("You should see a window with your camera feed")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Add text to frame
    cv2.putText(frame, "Camera Working! Press 'q' to continue", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Camera Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Test 4: Load emotion model
print("\n4. Testing emotion model...")
try:
    import pickle
    import mediapipe as mp
    
    MODEL_PATH = "/Users/joonyeoupkim/Desktop/prev_years/Spring_2025/Notre_Dame_Research/Gemma/advanced_emotion_model.pkl"
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    print("✅ Emotion model loaded!")
    print(f"   Model type: {type(model_data[0])}")
    print(f"   Emotions: {list(model_data[2].classes_)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Test 5: Simple emotion detection
print("\n5. Testing simple emotion detection...")
print("Press 'q' to quit")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Simple face detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        cv2.putText(frame, "Face Detected!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Face Detection Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

print("\n✅ All tests completed!")
print("\nIf camera didn't show:")
print("1. Check camera permissions in System Preferences")
print("2. Make sure no other app is using the camera")
print("3. Try restarting your terminal")
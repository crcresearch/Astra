import cv2
import mediapipe as mp
import numpy as np
import json
import os
import argparse
import pickle
from extract import PoseFeatureExtractor

def extract_pose_from_video(video_path, output_dir):
    """
    Extract pose data from a video using MediaPipe and save as JSON files
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save JSON files
    
    Returns:
        Path to directory containing JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0, 1, or 2 - higher is more accurate but slower
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    
    # Process each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(frame_rgb)
        
        # If pose was detected
        if results.pose_landmarks:
            # Convert MediaPipe format to OpenPose-like format (for compatibility)
            keypoints = []
            
            # MediaPipe has 33 landmarks, but we'll map the first 25 to match OpenPose
            # Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
            
            # Map MediaPipe landmarks to OpenPose format (approximate mapping)
            mp_to_op_mapping = {
                0: 0,    # nose
                11: 1,   # neck (using left shoulder)
                12: 2,   # right shoulder
                14: 3,   # right elbow
                16: 4,   # right wrist
                11: 5,   # left shoulder
                13: 6,   # left elbow
                15: 7,   # left wrist
                24: 8,   # mid hip
                24: 9,   # right hip
                26: 10,  # right knee
                28: 11,  # right ankle
                23: 12,  # left hip
                25: 13,  # left knee
                27: 14,  # left ankle
                2: 15,   # right eye
                5: 16,   # left eye
                7: 17,   # right ear
                8: 18,   # left ear
                31: 19,  # left big toe
                31: 20,  # left small toe (using big toe)
                29: 21,  # left heel
                32: 22,  # right big toe
                32: 23,  # right small toe (using big toe)
                30: 24   # right heel
            }
            
            # Convert from MediaPipe to OpenPose format
            openpose_data = {"people": [{"pose_keypoints_2d": []}]}
            flat_keypoints = []
            
            for op_idx in range(25):  # 25 keypoints in OpenPose
                mp_idx = mp_to_op_mapping.get(op_idx, 0)  # Default to nose if mapping not found
                
                landmark = results.pose_landmarks.landmark[mp_idx]
                
                # Get coordinates relative to image dimensions
                h, w, _ = frame.shape
                x = landmark.x * w
                y = landmark.y * h
                conf = landmark.visibility
                
                # Add to flat list
                flat_keypoints.extend([x, y, conf])
            
            # Add keypoints to OpenPose format
            openpose_data["people"][0]["pose_keypoints_2d"] = flat_keypoints
            
            # Save as JSON
            json_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'w') as f:
                json.dump(openpose_data, f)
        
        frame_idx += 1
        
        # Print progress
        if frame_idx % 10 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})", end='\r')
    
    # Release resources
    cap.release()
    pose.close()
    
    print(f"\nProcessed {frame_idx} frames, JSON files saved to {output_dir}")
    return output_dir

def predict_emotion(json_dir, model_path):
    """
    Predict emotion using extracted pose data
    
    Args:
        json_dir: Directory containing JSON files with pose data
        model_path: Path to trained model
        
    Returns:
        Predicted emotion and confidence
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, 0.0
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        if len(model_data) == 3:
            model, scaler, label_encoder = model_data
            print("Loaded model, scaler, and label encoder")
        elif len(model_data) == 2:
            model, scaler = model_data
            label_encoder = None
            print("Loaded model and scaler (no label encoder)")
        else:
            model = model_data
            scaler = None
            label_encoder = None
            print("Loaded model only (no scaler or label encoder)")
        
        # Extract features
        feature_extractor = PoseFeatureExtractor()
        features = feature_extractor.extract_features_from_json_files(json_dir)
        
        if features is None:
            print("Error: Could not extract features from JSON files")
            return None, 0.0
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if scaler is not None:
            features = scaler.transform(features)
        
        # Predict emotion
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Get prediction and confidence
        prediction = predictions[0]
        confidence = np.max(probabilities[0])
        
        # Convert prediction to emotion label
        if label_encoder is not None:
            emotion = label_encoder.inverse_transform([prediction])[0]
        else:
            # If no label encoder, use default mapping
            emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
            emotion = emotions[prediction] if prediction < len(emotions) else f"Unknown ({prediction})"
        
        return emotion, confidence
        
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def create_visualization(video_path, emotion, confidence, output_path=None):
    """
    Create a visualization of the video with the predicted emotion and face bounding boxes
    
    Args:
        video_path: Path to the original video
        emotion: Predicted emotion
        confidence: Prediction confidence
        output_path: Path to save the output video (optional)
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_emotion.mp4"
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for short-range detection (2 meters), 1 for full-range detection (5 meters)
        min_detection_confidence=0.5
    )
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define colors for each emotion (BGR format)
    emotion_colors = {
        'Anger': (0, 0, 255),       # Red
        'Disgust': (0, 255, 0),     # Green
        'Fear': (255, 0, 255),      # Magenta
        'Happiness': (0, 255, 255), # Yellow
        'Neutral': (255, 255, 255), # White
        'Sadness': (255, 0, 0),     # Blue
        'Surprise': (255, 255, 0)   # Cyan
    }
    
    # Use white if emotion not found in colors
    emotion_color = emotion_colors.get(emotion, (255, 255, 255))
    
    # Face bounding box color (bright green)
    bbox_color = (0, 255, 0)
    
    # Process each frame
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(frame_rgb)
        
        # Draw face bounding boxes
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                # Convert relative coordinates to pixel coordinates
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
                
                # Add detection confidence near the bounding box
                detection_confidence = detection.score[0]
                cv2.putText(frame, f"Face: {detection_confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
        
        # Add emotion text
        emotion_text = f"Emotion: {emotion} ({confidence:.2f})"
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
        
        # Add frame counter
        frame_text = f"Frame: {frame_count + 1}/{total_frames}"
        cv2.putText(frame, frame_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Print every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Visualization progress: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
    
    # Release resources
    cap.release()
    out.release()
    face_detection.close()
    
    print(f"\nVisualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detect emotions in videos using body pose with face bounding boxes")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", help="Path to save output video (optional)")
    parser.add_argument("--temp_dir", default="temp_json", help="Directory to save temporary JSON files")
    parser.add_argument("--keep_json", action="store_true", help="Keep JSON files after processing")
    
    args = parser.parse_args()
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    try:
        # Step 1: Extract pose data from video
        print("\n--- Step 1: Extracting pose data from video ---")
        json_dir = extract_pose_from_video(args.video, args.temp_dir)
        
        if not json_dir:
            print("Error: Failed to extract pose data from video")
            return
        
        # Step 2: Predict emotion using extracted pose data
        print("\n--- Step 2: Predicting emotion ---")
        emotion, confidence = predict_emotion(json_dir, args.model)
        
        if emotion:
            print(f"\nDetected emotion: {emotion}")
            print(f"Confidence: {confidence:.2f}")
            
            # Step 3: Create visualization with face bounding boxes
            print("\n--- Step 3: Creating visualization with face bounding boxes ---")
            create_visualization(args.video, emotion, confidence, args.output)
        
    finally:
        # Clean up temporary files
        if not args.keep_json:
            import shutil
            shutil.rmtree(args.temp_dir)
            print(f"Removed temporary directory: {args.temp_dir}")

if __name__ == "__main__":
    main()

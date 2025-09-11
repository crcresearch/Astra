import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import argparse
import os

class RealTimeEmotionDetector:
    def __init__(self, model_path, classes_path=None):
        """
        Initialize the real-time emotion detector (OpenCV only version)
        
        Args:
            model_path: Path to the trained emotion recognition model
            classes_path: Path to the class names file
        """
        self.model_path = model_path
        
        # Load the trained model
        print(f"Loading emotion recognition model from {model_path}...")
        self.model = load_model(model_path)
        
        # Load class names
        if classes_path is None:
            classes_path = os.path.splitext(model_path)[0] + "_classes.txt"
        
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # Default emotion classes if file not found
            self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        print(f"Loaded {len(self.class_names)} emotion classes: {self.class_names}")
        
        # Initialize OpenCV face detection
        self.setup_face_detection()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Emotion colors (BGR format for OpenCV)
        self.emotion_colors = {
            'Angry': (0, 0, 255),       # Red
            'Disgust': (0, 255, 0),     # Green
            'Fear': (255, 0, 255),      # Magenta
            'Happy': (0, 255, 255),     # Yellow
            'Neutral': (255, 255, 255), # White
            'Sad': (255, 0, 0),         # Blue
            'Surprise': (255, 255, 0),  # Cyan
        }
        
        # Smoothing for emotion predictions
        self.emotion_history = []
        self.history_length = 5  # Number of frames to average
        
        # Last detected faces for tracking
        self.last_faces = []
    
    def setup_face_detection(self):
        """
        Setup OpenCV face detection
        """
        # Load OpenCV's face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
        
        print("Using OpenCV face detection")
    
    def detect_faces(self, frame):
        """
        Detect faces using OpenCV
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with multiple scale factors for better results
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def preprocess_face(self, face, target_size=(160, 160)):
        """
        Preprocess face for emotion recognition
        
        Args:
            face: Cropped face image
            target_size: Target size for the model
            
        Returns:
            Preprocessed face tensor
        """
        try:
            # Resize face
            face_resized = cv2.resize(face, target_size)
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to array and normalize
            face_array = img_to_array(face_rgb)
            
            # Preprocess based on the model type (assuming MobileNetV2)
            face_array = tf.keras.applications.mobilenet_v2.preprocess_input(face_array)
            
            # Add batch dimension
            face_array = np.expand_dims(face_array, axis=0)
            
            return face_array
        
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_emotion(self, face):
        """
        Predict emotion for a single face
        
        Args:
            face: Cropped face image
            
        Returns:
            emotion, confidence
        """
        # Preprocess face
        face_tensor = self.preprocess_face(face)
        
        if face_tensor is None:
            return "Unknown", 0.0
        
        try:
            # Predict emotion
            predictions = self.model.predict(face_tensor, verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            emotion = self.class_names[predicted_idx]
            
            return emotion, confidence
        
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "Error", 0.0
    
    def smooth_emotion_prediction(self, emotion, confidence):
        """
        Smooth emotion predictions over multiple frames
        
        Args:
            emotion: Current emotion prediction
            confidence: Current confidence
            
        Returns:
            Smoothed emotion and confidence
        """
        # Add current prediction to history
        self.emotion_history.append((emotion, confidence))
        
        # Keep only recent history
        if len(self.emotion_history) > self.history_length:
            self.emotion_history.pop(0)
        
        # If we have enough history, use weighted average
        if len(self.emotion_history) >= 3:
            # Count emotions in history with weights (recent frames have higher weight)
            emotion_scores = {}
            total_weight = 0
            
            for i, (hist_emotion, hist_confidence) in enumerate(self.emotion_history):
                weight = (i + 1) / len(self.emotion_history)  # Linear weight increase
                weighted_confidence = hist_confidence * weight
                
                if hist_emotion in emotion_scores:
                    emotion_scores[hist_emotion] += weighted_confidence
                else:
                    emotion_scores[hist_emotion] = weighted_confidence
                
                total_weight += weight
            
            # Get most confident emotion
            if emotion_scores:
                best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                avg_confidence = best_emotion[1] / total_weight
                return best_emotion[0], min(avg_confidence, 1.0)
        
        return emotion, confidence
    
    def update_fps(self):
        """
        Update FPS counter
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence):
        """
        Draw emotion information on the frame
        
        Args:
            frame: Input frame
            x, y, w, h: Face bounding box
            emotion: Predicted emotion
            confidence: Prediction confidence
        """
        # Get color for this emotion
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw face bounding box with thickness based on confidence
        thickness = max(2, int(confidence * 4))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Prepare text
        text = f"{emotion}: {confidence:.2f}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # Draw text background with some padding
        padding = 5
        bg_x1 = x
        bg_y1 = y - text_height - padding * 2
        bg_x2 = x + text_width + padding * 2
        bg_y2 = y
        
        # Ensure background rectangle is within frame bounds
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(frame.shape[1], bg_x2)
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Draw text
        text_x = x + padding
        text_y = y - padding
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness)
        
        # Draw confidence bar below the face
        bar_width = w
        bar_height = 6
        bar_x = x
        bar_y = y + h + 5
        
        # Ensure bar is within frame bounds
        if bar_y + bar_height < frame.shape[0]:
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            
            # Confidence bar
            conf_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), color, -1)
    
    def run_camera(self, camera_index=0, save_video=False, output_path="emotion_detection.mp4", 
                   show_fps=True, frame_skip=1):
        """
        Run real-time emotion detection on camera feed
        
        Args:
            camera_index: Camera index (0 for default camera)
            save_video: Whether to save the output video
            output_path: Path to save the output video
            show_fps: Whether to show FPS on screen
            frame_skip: Process every nth frame (for performance)
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            # Try different camera indices
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Found camera at index {i}")
                    break
            else:
                raise RuntimeError(f"Cannot open any camera (tried indices 0-4)")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
        
        # Get actual camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {width}x{height} @ {fps}fps")
        
        # Initialize video writer if saving
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            print(f"Saving video to {output_path}")
        
        frame_count = 0
        print("\n" + "="*50)
        print("🎭 REAL-TIME EMOTION DETECTION STARTED")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reset emotion history")
        print("="*50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame_count += 1
                
                # Process every nth frame for performance
                if frame_count % frame_skip == 0:
                    # Detect faces
                    faces = self.detect_faces(frame)
                    
                    # Process each detected face
                    for (x, y, w, h) in faces:
                        # Ensure coordinates are within frame bounds
                        x, y = max(0, x), max(0, y)
                        w, h = min(w, width - x), min(h, height - y)
                        
                        if w > 60 and h > 60:  # Only process reasonably sized faces
                            # Extract face region with some padding
                            padding = 10
                            face_x1 = max(0, x - padding)
                            face_y1 = max(0, y - padding)
                            face_x2 = min(width, x + w + padding)
                            face_y2 = min(height, y + h + padding)
                            
                            face = frame[face_y1:face_y2, face_x1:face_x2]
                            
                            if face.size > 0:
                                # Predict emotion
                                emotion, confidence = self.predict_emotion(face)
                                
                                # Only show predictions with reasonable confidence
                                if confidence > 0.3:
                                    # Smooth prediction
                                    emotion, confidence = self.smooth_emotion_prediction(emotion, confidence)
                                    
                                    # Draw emotion information
                                    self.draw_emotion_info(frame, x, y, w, h, emotion, confidence)
                                else:
                                    # Draw just the face box for low confidence
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)
                                    cv2.putText(frame, "Low confidence", (x, y-5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                # Update FPS
                if show_fps:
                    self.update_fps()
                    cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save frame if recording
                if out:
                    out.write(frame)
                
                # Display frame
                cv2.imshow('Real-Time Emotion Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    screenshot_path = f"emotion_screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Reset emotion history
                    self.emotion_history = []
                    print("🔄 Emotion history reset")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error during detection: {e}")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("🔧 Camera released and windows closed")
            print(f"📊 Total frames processed: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Real-time facial emotion detection (OpenCV only)')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained emotion recognition model')
    parser.add_argument('--classes_path', type=str, default=None,
                       help='Path to class names file')
    parser.add_argument('--camera_index', type=int, default=0,
                       help='Camera index (0 for default camera)')
    parser.add_argument('--save_video', action='store_true',
                       help='Save output video')
    parser.add_argument('--output_path', type=str, default='emotion_detection.mp4',
                       help='Output video path')
    parser.add_argument('--frame_skip', type=int, default=2,
                       help='Process every nth frame (higher = faster but less accurate)')
    parser.add_argument('--no_fps', action='store_true',
                       help='Hide FPS counter')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Initialize detector
    detector = RealTimeEmotionDetector(
        model_path=args.model_path,
        classes_path=args.classes_path
    )
    
    # Run real-time detection
    detector.run_camera(
        camera_index=args.camera_index,
        save_video=args.save_video,
        output_path=args.output_path,
        show_fps=not args.no_fps,
        frame_skip=args.frame_skip
    )

if __name__ == "__main__":
    main()

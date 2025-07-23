#!/usr/bin/env python3
# realtimeinfer_heart_gaze_cuda.py
"""
everything with face, body, heart, audio, gaze
download everything from Requirements.txt and run realtimeinfer.py
"""

import os
import sys
import time
import traceback
import json
import datetime
import threading
import queue
from collections import deque
import numpy as np
from scipy import signal as scipy_signal

import cv2
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pickle

# ===== CUDA CONFIGURATION =====
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available, using CPU")

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SiNC-rPPG-webcam', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '3DGazeNet'))

# Import heart rate utilities
from utils_heart.model_selector import select_model

# Import gaze tracking utilities
from models_gaze import FaceDetectorIF as FaceDetector
from models_gaze import GazePredictorHandler as GazePredictor
from utils_gaze import config as gaze_cfg, update_config, draw_results

# Audio backend
try:
    import sounddevice as sd
    audio_backend = 'sounddevice'
except ImportError:
    try:
        import pyaudio
        audio_backend = 'pyaudio'
    except ImportError:
        print("⚠️ No audio backend found. Audio features will be disabled.")
        audio_backend = None

# Suppress logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

print("=== Multimodal Emotion + Heart Rate + Gaze Detection System ===")

# ===== CONFIGURATION =====
# Face emotion model
FACE_MODEL_PATH = "emotion_model_fixed.h5"

# Audio models
AUDIO_MODELS_DIR = os.path.join(os.path.dirname(__file__), "patient_models_balanced")
DEFAULT_PATIENT = "P01"

# Heart rate model
HR_MODEL_PATH = "experiments/fold0/physnet"

# Gaze model config
GAZE_CONFIG_PATH = "configs_gaze/infer_res18_x128_all_vfhq_vert.yaml"

# Audio settings
SAMPLE_RATE = 16000
N_MELS = 128
AUDIO_WINDOW_SIZE = 2.0
AUDIO_PROCESS_INTERVAL = 0.5
MIN_AUDIO_ENERGY = 0.01

# Video settings
VIDEO_FPS_TARGET = 30

# Modality weights
FACE_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4

# Face emotions
face_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ===== AUDIO MODEL DEFINITION =====
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.3)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout_fc = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x1 = torch.relu(self.fc1(x))
        x1 = self.dropout_fc(x1)
        
        x2 = torch.relu(self.fc2(x1))
        x2 = self.dropout_fc(x2)
        
        out = self.fc3(x2)
        
        return out

# ===== GAZE PROCESSOR =====
class GazeProcessor:
    """Process eye gaze tracking"""
    
    def __init__(self, config_path, smoothing_frames=3):
        # Update gaze config
        update_config(config_path)
        
        # Initialize models
        self.detector = FaceDetector(gaze_cfg.DETECTOR.THRESHOLD, gaze_cfg.DETECTOR.IMAGE_SIZE)
        self.predictor = GazePredictor(gaze_cfg.PREDICTOR, device=DEVICE if 'cuda' in str(DEVICE) else 'cpu')
        
        # Smoothing
        self.smoothing_frames = smoothing_frames
        self.gaze_buffer = deque(maxlen=smoothing_frames)
        
        # Latest gaze data
        self.latest_gaze = None
        self.latest_landmarks = None
        self.lock = threading.Lock()
        
        print("✅ Gaze processor initialized")
    
    def smooth_gaze(self, gaze_vector):
        """Apply temporal smoothing to gaze predictions"""
        self.gaze_buffer.append(gaze_vector)
        if len(self.gaze_buffer) > 1:
            smoothed = np.mean(self.gaze_buffer, axis=0)
            return smoothed / np.linalg.norm(smoothed)
        return gaze_vector
    
    def process_frame(self, frame):
        """Process frame for gaze detection"""
        try:
            # Run face detection
            bboxes, lms5, _ = self.detector.run(frame)
            
            if bboxes is not None and len(bboxes) > 0:
                # Get largest face
                idxs_sorted = sorted(range(len(bboxes)), key=lambda k: bboxes[k][3] - bboxes[k][1])
                lms5_largest = lms5[idxs_sorted[-1]]
                
                # Run gaze prediction
                out_dict = self.predictor(frame, lms5_largest, undo_roll=True)
                
                if out_dict is not None:
                    # Apply smoothing
                    smoothed_gaze = self.smooth_gaze(out_dict['gaze_out'])
                    out_dict['gaze_out'] = smoothed_gaze
                    
                    with self.lock:
                        self.latest_gaze = out_dict
                        self.latest_landmarks = lms5_largest
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"Gaze processing error: {e}")
            return False
    
    def get_latest_gaze(self):
        """Get latest gaze data"""
        with self.lock:
            return self.latest_gaze, self.latest_landmarks
    
    def draw_gaze(self, frame):
        """Draw gaze visualization on frame"""
        gaze_data, landmarks = self.get_latest_gaze()
        if gaze_data is not None and landmarks is not None:
            return draw_results(frame, landmarks, gaze_data)
        return frame

# ===== HEART RATE PROCESSOR =====
class HeartRateProcessor:
    """Heart rate processing"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = DEVICE
        self.model = self.load_hr_model(model_path)
        
        # Buffers
        self.frame_buffer = deque(maxlen=150)
        self.heart_rates = deque(maxlen=20)
        self.confidence_scores = deque(maxlen=20)
        
        # Latest values
        self.latest_hr = None
        self.latest_confidence = 0.0
        self.lock = threading.Lock()
        
        print("✅ Heart rate processor initialized")
    
    def load_hr_model(self, model_path):
        """Load PhysNet model"""
        print(f"Loading heart rate model from: {model_path}")
        
        class CompleteArgs:
            def __init__(self):
                self.model_type = 'physnet'
                self.channels = 'rgb'
                self.frame_height = 64
                self.frame_width = 64
                self.dropout = 0.5
                self.fps = 30
                self.fpc = 150
        
        args = CompleteArgs()
        model = select_model(args)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)
        
        print("✅ Heart rate model loaded")
        return model
    
    def add_frame(self, face_crop):
        """Add face frame to buffer"""
        if face_crop is None:
            return
        
        face_resized = cv2.resize(face_crop, (64, 64))
        
        face_yuv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YUV)
        face_yuv[:,:,0] = cv2.equalizeHist(face_yuv[:,:,0])
        face_normalized = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
        
        with self.lock:
            self.frame_buffer.append(face_normalized)
    
    def process_heart_rate(self):
        """Process buffered frames"""
        with self.lock:
            if len(self.frame_buffer) < 150:
                return None, 0.0
            
            frames = np.array(list(self.frame_buffer))
        
        try:
            frames = frames.astype(np.float32) / 255.0
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
            frames = frames.to(DEVICE)
            
            with torch.no_grad():
                prediction = self.model(frames)
                signal = prediction.cpu().numpy().flatten()
            
            hr, confidence = self.signal_to_heart_rate(signal)
            
            with self.lock:
                if hr and confidence > 0.3:
                    self.heart_rates.append(hr)
                    self.confidence_scores.append(confidence)
                    self.latest_hr = hr
                    self.latest_confidence = confidence
            
            return hr, confidence
            
        except Exception as e:
            print(f"HR processing error: {e}")
            return None, 0.0
    
    def signal_to_heart_rate(self, signal, fps=30):
        """Convert signal to heart rate"""
        try:
            nyquist = fps / 2
            low_freq = 0.7 / nyquist
            high_freq = 4.0 / nyquist
            
            b, a = scipy_signal.butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = scipy_signal.filtfilt(b, a, signal)
            
            windowed_signal = filtered_signal * scipy_signal.windows.hann(len(filtered_signal))
            
            freqs = np.fft.rfftfreq(len(windowed_signal), 1/fps)
            fft = np.abs(np.fft.rfft(windowed_signal))
            
            valid_idx = (freqs >= 0.7) & (freqs <= 4.0)
            
            if not np.any(valid_idx):
                return 75, 0.0
            
            valid_freqs = freqs[valid_idx]
            valid_fft = fft[valid_idx]
            
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            peak_power = valid_fft[peak_idx]
            
            mean_power = np.mean(valid_fft)
            confidence = min(1.0, peak_power / (mean_power * 3))
            
            heart_rate = peak_freq * 60
            
            if heart_rate < 50 or heart_rate > 180:
                confidence *= 0.5
            
            return heart_rate, confidence
            
        except Exception as e:
            print(f"HR calculation error: {e}")
            return 75, 0.0
    
    def get_latest_hr(self):
        """Get latest heart rate"""
        with self.lock:
            if self.latest_hr and len(self.heart_rates) > 0:
                recent_hrs = list(self.heart_rates)[-3:]
                smoothed_hr = np.mean(recent_hrs)
                return smoothed_hr, self.latest_confidence
            return None, 0.0

# ===== AUDIO PROCESSOR =====
class AudioProcessor:
    def __init__(self, models_dir, patient_id):
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * AUDIO_WINDOW_SIZE))
        self.latest_emotion = None
        self.latest_confidence = 0.0
        self.latest_probs = None
        self.is_running = False
        self.lock = threading.Lock()
        
        self.audio_model = None
        self.audio_encoder = None
        self.audio_enabled = False
        
        self.load_audio_model(models_dir, patient_id)
        
        self.audio_stats = {
            'samples_processed': 0,
            'predictions_made': 0,
            'last_energy': 0.0,
            'last_update': time.time()
        }
    
    def load_audio_model(self, models_dir, patient_id):
        patient_model_dir = os.path.join(models_dir, patient_id)
        
        if not os.path.exists(os.path.join(patient_model_dir, "model.pt")):
            print(f"⚠️ No audio model found for patient {patient_id}")
            return
        
        try:
            with open(os.path.join(patient_model_dir, "encoder.pkl"), "rb") as f:
                self.audio_encoder = pickle.load(f)
            
            self.audio_model = ImprovedCNN(num_classes=len(self.audio_encoder.classes_))
            checkpoint = torch.load(
                os.path.join(patient_model_dir, "model.pt"),
                map_location=DEVICE,
                weights_only=False
            )
            self.audio_model.load_state_dict(checkpoint['model_state_dict'])
            self.audio_model.eval()
            self.audio_model.to(DEVICE)
            
            self.audio_enabled = True
            print(f"✅ Audio model loaded for patient {patient_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to load audio model: {e}")
            self.audio_enabled = False
    
    def start(self):
        if not self.audio_enabled or audio_backend is None:
            print("⚠️ Audio processing disabled")
            return
        
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        
        if audio_backend == 'sounddevice':
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback_sd,
                blocksize=1024
            )
            self.stream.start()
            
        elif audio_backend == 'pyaudio':
            import pyaudio
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback_pa
            )
            self.stream.start_stream()
        
        print("🎤 Audio processing started")
    
    def stop(self):
        self.is_running = False
        
        if hasattr(self, 'stream') and self.stream:
            if audio_backend == 'pyaudio':
                self.stream.stop_stream()
                self.stream.close()
                if hasattr(self, 'pyaudio_instance'):
                    self.pyaudio_instance.terminate()
            elif audio_backend == 'sounddevice':
                self.stream.stop()
                self.stream.close()
    
    def _audio_callback_sd(self, indata, frames, time_info, status):
        audio_data = indata[:, 0].copy()
        with self.lock:
            self.audio_buffer.extend(audio_data)
            self.audio_stats['samples_processed'] += len(audio_data)
    
    def _audio_callback_pa(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        with self.lock:
            self.audio_buffer.extend(audio_data)
            self.audio_stats['samples_processed'] += len(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _audio_loop(self):
        last_process_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_process_time >= AUDIO_PROCESS_INTERVAL:
                with self.lock:
                    if len(self.audio_buffer) >= int(SAMPLE_RATE * 0.5):
                        audio_chunk = np.array(list(self.audio_buffer)[-int(SAMPLE_RATE * 1.0):])
                        energy = np.sqrt(np.mean(audio_chunk**2))
                        self.audio_stats['last_energy'] = energy
                        
                        if energy > MIN_AUDIO_ENERGY:
                            self._process_audio(audio_chunk)
                            self.audio_stats['last_update'] = current_time
                
                last_process_time = current_time
            
            time.sleep(0.05)
    
    def _process_audio(self, audio):
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            if len(audio_trimmed) < int(SAMPLE_RATE * 0.1):
                return
            
            segment_length = int(SAMPLE_RATE * 0.5)
            if len(audio_trimmed) > segment_length:
                start = (len(audio_trimmed) - segment_length) // 2
                audio_trimmed = audio_trimmed[start:start + segment_length]
            else:
                audio_trimmed = np.pad(audio_trimmed, (0, segment_length - len(audio_trimmed)), mode='constant')
            
            mel = librosa.feature.melspectrogram(
                y=audio_trimmed, sr=SAMPLE_RATE, n_mels=N_MELS,
                n_fft=1024, hop_length=256, fmax=8000
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db + 80) / 80
            
            x = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()
            x = x.to(DEVICE)
            
            with torch.no_grad():
                logits = self.audio_model(x)
                probs = F.softmax(logits, dim=1)
                
                audio_probs = {}
                for idx, emotion in enumerate(self.audio_encoder.classes_):
                    audio_probs[emotion] = probs[0, idx].item()
                
                top_emotion = max(audio_probs, key=audio_probs.get)
                top_confidence = audio_probs[top_emotion]
                
                with self.lock:
                    self.latest_emotion = top_emotion
                    self.latest_confidence = top_confidence
                    self.latest_probs = audio_probs
                    self.audio_stats['predictions_made'] += 1
                
        except Exception as e:
            print(f"Audio processing error: {e}")
    
    def get_latest_prediction(self):
        with self.lock:
            if self.latest_emotion and (time.time() - self.audio_stats['last_update'] < 2.0):
                return self.latest_emotion, self.latest_confidence, self.latest_probs
            return None, 0.0, None

# ===== COMBINED DETECTOR =====
class MultimodalBiometricDetector:
    def __init__(self, face_model_path, audio_models_dir, hr_model_path, gaze_config_path, patient_id=DEFAULT_PATIENT):
        # Load face emotion model
        print(f"Loading face emotion model...")
        self.face_model = tf.keras.models.load_model(face_model_path, compile=False)
        self.face_model.compile(optimizer="adam", loss="categorical_crossentropy")
        print("✅ Face model ready!")
        
        # Initialize processors
        self.audio_processor = AudioProcessor(audio_models_dir, patient_id)
        self.hr_processor = HeartRateProcessor(hr_model_path)
        self.gaze_processor = GazeProcessor(gaze_config_path)
        
        # Modality weights
        self.face_weight = FACE_WEIGHT
        self.audio_weight = AUDIO_WEIGHT
        
        # Emotion mapping
        self.emotion_mapping = {
            'Angry': ['dysregulated', 'frustrated', 'protest', 'angry'],
            'Happy': ['happy', 'delighted', 'glee', 'laughter', 'joy'],
            'Sad': ['sad', 'cry', 'upset'],
            'Fear': ['dysregulated', 'help', 'upset'],
            'Surprise': ['excited', 'glee'],
            'Disgust': ['protest', 'no'],
            'Neutral': ['neutral', 'social', 'yes', 'no']
        }
        
        # Processing intervals
        self.last_hr_process = time.time()
        self.hr_process_interval = 2.0
    
    def start_audio(self):
        self.audio_processor.start()
    
    def stop_audio(self):
        self.audio_processor.stop()
    
    def preprocess_face(self, face):
        face = cv2.resize(face, (160, 160))
        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)
        else:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        return np.expand_dims(face, axis=0)
    
    def predict_face_emotion(self, face):
        inp = self.preprocess_face(face)
        preds = self.face_model.predict(inp, verbose=0)[0]
        
        probs = {}
        for i, emotion in enumerate(face_emotions):
            probs[emotion] = float(preds[i])
        
        return probs
    
    def map_audio_to_face_emotions(self, audio_probs):
        face_probs = {emotion: 0.0 for emotion in face_emotions}
        
        for face_emotion, audio_emotions in self.emotion_mapping.items():
            total_prob = 0.0
            count = 0
            
            for audio_emotion in audio_emotions:
                for audio_key, prob in audio_probs.items():
                    if audio_emotion in audio_key.lower() or audio_key.lower() in audio_emotion:
                        total_prob += prob
                        count += 1
            
            if count > 0:
                face_probs[face_emotion] = total_prob / count
        
        total = sum(face_probs.values())
        if total > 0:
            for emotion in face_probs:
                face_probs[emotion] /= total
        
        return face_probs
    
    def fuse_emotions(self, face_probs, audio_emotion=None, audio_confidence=0.0, audio_probs=None):
        if audio_emotion is None or audio_probs is None:
            return face_probs, 1.0, 0.0
        
        mapped_audio_probs = self.map_audio_to_face_emotions(audio_probs)
        
        fused_probs = {}
        for emotion in face_emotions:
            face_p = face_probs.get(emotion, 0.0)
            audio_p = mapped_audio_probs.get(emotion, 0.0)
            fused_probs[emotion] = (self.face_weight * face_p + self.audio_weight * audio_p)
        
        return fused_probs, self.face_weight, self.audio_weight
    
    def add_face_for_hr(self, face_crop):
        """Add face crop to heart rate processor"""
        self.hr_processor.add_frame(face_crop)
    
    def process_heart_rate(self):
        """Process heart rate if enough time has passed"""
        current_time = time.time()
        if current_time - self.last_hr_process > self.hr_process_interval:
            hr, confidence = self.hr_processor.process_heart_rate()
            self.last_hr_process = current_time
            return hr, confidence
        return None, 0.0
    
    def process_gaze(self, frame):
        """Process gaze tracking"""
        return self.gaze_processor.process_frame(frame)

# Enhanced feedback templates
feedback_templates = {
    'Angry': {
        'normal_hr': "Patient frustrated. Use calming strategies: deep pressure, quiet space, sensory tools.",
        'high_hr': "Patient frustrated with elevated heart rate. Priority: calming techniques, breathing exercises, reduce stimulation.",
        'low_hr': "Patient frustrated but calm physically. Address emotional needs with validation and redirection."
    },
    'Happy': {
        'normal_hr': "Positive state! Reinforce with preferred activities, social stories, or special interests.",
        'high_hr': "Excited and happy! Monitor for overstimulation, provide structured activities to channel energy.",
        'low_hr': "Content and calm. Excellent time for learning new skills or social interaction."
    },
    'Fear': {
        'normal_hr': "Patient fearful. Provide comfort items, use reassuring tone, maintain familiar routines.",
        'high_hr': "Patient anxious with stress response. Immediate comfort needed, remove triggers, use grounding techniques.",
        'low_hr': "Mild concern detected. Preventive comfort measures and reassurance recommended."
    },
    'Sad': {
        'normal_hr': "Patient appears down. Offer comfort, validate feelings, use visual supports.",
        'high_hr': "Distressed and sad. Physical comfort, sensory regulation, and emotional support needed.",
        'low_hr': "Low mood detected. Gentle engagement with preferred activities may help."
    },
    'Disgust': {
        'normal_hr': "Patient showing aversion. Reduce sensory triggers, offer preferred alternatives.",
        'high_hr': "Strong aversion with stress response. Remove trigger immediately, provide sensory break.",
        'low_hr': "Mild discomfort. Adjust environment or activity as needed."
    },
    'Surprise': {
        'normal_hr': "Unexpected reaction. Provide clear explanations, visual schedules, return to routine.",
        'high_hr': "Startled response. Allow recovery time, explain changes, use calming strategies.",
        'low_hr': "Mild surprise. Continue with gentle explanation and reassurance."
    },
    'Neutral': {
        'normal_hr': "Calm state. Good for introducing new activities with visual supports.",
        'high_hr': "Alert but neutral. May be processing internally, allow time before new activities.",
        'low_hr': "Very calm state. Optimal for rest or quiet activities."
    }
}

def generate_enhanced_feedback(emotion, confidence, heart_rate=None, hr_confidence=0.0, gaze_data=None):
    """Generate feedback considering emotion, heart rate, and gaze"""
    
    # Determine HR category
    hr_category = 'normal_hr'
    if heart_rate:
        if heart_rate > 100:
            hr_category = 'high_hr'
        elif heart_rate < 60:
            hr_category = 'low_hr'
    
    # Get base feedback
    if emotion in feedback_templates:
        base_feedback = feedback_templates[emotion].get(hr_category, feedback_templates[emotion]['normal_hr'])
    else:
        base_feedback = "Monitor patient closely."
    
    # Add gaze-specific guidance if available
    if gaze_data:
        gaze_vec = gaze_data.get('gaze_out', [0, 0, 0])
        # Check if looking away (not at camera/caregiver)
        if abs(gaze_vec[0]) > 0.3 or abs(gaze_vec[1]) > 0.3:
            base_feedback += " Patient avoiding eye contact - respect their comfort level."
    
    return base_feedback

# Create filename for feedback
session_start = datetime.datetime.now()
timestamp = session_start.strftime("%Y%m%d_%H%M%S")
feedback_filename = f"multimodal_biometric_gaze_feedbacks_{timestamp}.json"

def save_feedback_entry(entry):
    try:
        if not os.path.exists(feedback_filename):
            with open(feedback_filename, 'w') as f:
                f.write("[\n")
        
        needs_comma = False
        if os.path.getsize(feedback_filename) > 2:
            needs_comma = True
        
        with open(feedback_filename, 'a') as f:
            if needs_comma:
                f.write(",\n")
            json.dump(entry, f, indent=2)
    except Exception as e:
        print(f"Failed to save feedback entry: {e}")

def finalize_feedback_file():
    try:
        if os.path.exists(feedback_filename):
            with open(feedback_filename, 'a') as f:
                f.write("\n]")
            print(f"Finalized feedback file: {feedback_filename}")
    except Exception as e:
        print(f"Failed to finalize feedback file: {e}")

# ===== MAIN LOOP =====
print("\n🚀 Starting Multimodal Biometric System with Gaze Tracking")
print(f"Monitoring: Emotions (Face + Audio) + Heart Rate + Eye Gaze")
print(f"Weights: Face {FACE_WEIGHT:.0%}, Audio {AUDIO_WEIGHT:.0%}")
print(f"Feedback saved to: {feedback_filename}")

# Initialize detector
detector = MultimodalBiometricDetector(
    face_model_path=FACE_MODEL_PATH,
    audio_models_dir=AUDIO_MODELS_DIR,
    hr_model_path=HR_MODEL_PATH,
    gaze_config_path=GAZE_CONFIG_PATH,
    patient_id=DEFAULT_PATIENT
)

# Start audio
detector.start_audio()

# Face detector for emotion
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open camera
print("\nOpening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    detector.stop_audio()
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS_TARGET)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera ready — press 'q' to quit")
print("Press 'w' to adjust weights")
print("Press 'h' to toggle heart rate display")
print("Press 'g' to toggle gaze display\n")

# Tracking variables
last_feedback = time.time()
current_feedback = "Starting..."
feedback_interval = 60.0

# Latest values
last_emotion = None
last_confidence = 0.0
last_face_emotion = None
last_face_confidence = 0.0
last_audio_emotion = None
last_audio_confidence = 0.0
last_heart_rate = None
last_hr_confidence = 0.0
last_gaze_data = None
face_weight_used = 1.0
audio_weight_used = 0.0

# Display settings
show_hr_details = True
show_gaze_details = True

# Frame timing
frame_time = 1.0 / VIDEO_FPS_TARGET
last_frame_time = time.time()

# Create initial feedback file
if not os.path.exists(feedback_filename):
    with open(feedback_filename, 'w') as f:
        f.write("[\n")

try:
    while True:
        # Frame rate control
        current_time = time.time()
        if current_time - last_frame_time < frame_time:
            continue
        last_frame_time = current_time
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        display = frame.copy()
        H, W = display.shape[:2]
        
        # Process gaze first (it has its own face detection)
        gaze_detected = detector.process_gaze(frame)
        if gaze_detected:
            gaze_data, _ = detector.gaze_processor.get_latest_gaze()
            last_gaze_data = gaze_data
        
        # Draw gaze visualization if enabled
        if show_gaze_details and gaze_detected:
            display = detector.gaze_processor.draw_gaze(display)
        
        # Detect faces for emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Process largest face for emotion and HR
        if len(faces) > 0:
            face_bbox = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face_bbox
            
            # Extract face for processing
            face_crop = frame[y:y+h, x:x+w]
            
            # Add to heart rate buffer
            detector.add_face_for_hr(face_crop)
            
            # Get face emotion
            face_probs = detector.predict_face_emotion(face_crop)
            last_face_emotion = max(face_probs, key=face_probs.get)
            last_face_confidence = face_probs[last_face_emotion]
            
            # Get audio emotion
            audio_em, audio_conf, audio_probs = detector.audio_processor.get_latest_prediction()
            
            if audio_em:
                last_audio_emotion = audio_em
                last_audio_confidence = audio_conf
            else:
                last_audio_emotion = None
                last_audio_confidence = 0.0
            
            # Fuse emotions
            fused_probs, face_w, audio_w = detector.fuse_emotions(
                face_probs, last_audio_emotion, last_audio_confidence, audio_probs
            )
            
            last_emotion = max(fused_probs, key=fused_probs.get)
            last_confidence = fused_probs[last_emotion]
            face_weight_used = face_w
            audio_weight_used = audio_w
            
            # Process heart rate
            hr, hr_conf = detector.process_heart_rate()
            if hr:
                last_heart_rate = hr
                last_hr_confidence = hr_conf
                print(f"💓 HR: {hr:.1f} BPM (confidence: {hr_conf:.2f})")
            
            # Draw face box for emotion
            color = (0, 255, 0) if last_hr_confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            
            # Show emotion
            label = f"{last_emotion} {last_confidence:.0%}"
            if audio_em:
                label += f" (F:{face_w:.0%}/A:{audio_w:.0%})"
            cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Show modality details
            y_offset = y + h + 20
            cv2.putText(display, f"Face: {last_face_emotion} ({last_face_confidence:.0%})", 
                       (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            if last_audio_emotion:
                y_offset += 20
                cv2.putText(display, f"Audio: {last_audio_emotion} ({last_audio_confidence:.0%})", 
                           (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            if last_heart_rate and show_hr_details:
                y_offset += 20
                cv2.putText(display, f"HR: {last_heart_rate:.0f} BPM", 
                           (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        
        # Show gaze info in corner if detected
        if last_gaze_data and show_gaze_details:
            gaze_vec = last_gaze_data.get('gaze_out', [0, 0, 0])
            gaze_text = f"Gaze: ({gaze_vec[0]:.2f}, {gaze_vec[1]:.2f}, {gaze_vec[2]:.2f})"
            cv2.putText(display, gaze_text, (W-300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Periodic feedback update
        now = time.time()
        if now - last_feedback > feedback_interval:
            if last_emotion:
                # Generate feedback with all modalities
                current_feedback = generate_enhanced_feedback(
                    last_emotion, last_confidence, last_heart_rate, 
                    last_hr_confidence, last_gaze_data
                )
                
                # Create comprehensive feedback entry
                feedback_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "fused_emotion": last_emotion,
                    "fused_confidence": last_confidence,
                    "face_emotion": last_face_emotion,
                    "face_confidence": last_face_confidence,
                    "audio_emotion": last_audio_emotion,
                    "audio_confidence": last_audio_confidence,
                    "heart_rate": last_heart_rate,
                    "heart_rate_confidence": last_hr_confidence,
                    "gaze_vector": last_gaze_data.get('gaze_out', None).tolist() if last_gaze_data else None,
                    "face_weight": face_weight_used,
                    "audio_weight": audio_weight_used,
                    "modalities_active": {
                        "face": last_face_emotion is not None,
                        "audio": last_audio_emotion is not None,
                        "heart_rate": last_heart_rate is not None,
                        "gaze": last_gaze_data is not None
                    },
                    "feedback": current_feedback
                }
                
                save_feedback_entry(feedback_entry)
                
                # Print summary
                print(f"\n[{feedback_entry['timestamp']}]")
                print(f"  Emotion - Fused: {last_emotion} ({last_confidence:.0%})")
                print(f"  Face: {last_face_emotion} ({last_face_confidence:.0%})")
                if last_audio_emotion:
                    print(f"  Audio: {last_audio_emotion} ({last_audio_confidence:.0%})")
                if last_heart_rate:
                    print(f"  Heart Rate: {last_heart_rate:.0f} BPM (conf: {last_hr_confidence:.0%})")
                if last_gaze_data:
                    gaze_vec = last_gaze_data.get('gaze_out', [0, 0, 0])
                    print(f"  Gaze: ({gaze_vec[0]:.2f}, {gaze_vec[1]:.2f}, {gaze_vec[2]:.2f})")
                print(f"  Feedback: {current_feedback}")
            else:
                current_feedback = "No face detected"
                
            last_feedback = now
        
        # UI Overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0,0), (W,140), (0,0,0), -1)
        cv2.rectangle(overlay, (0,H-80), (W,H), (0,0,0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
        
        # Title
        cv2.putText(display, "Multimodal Biometric + Gaze Monitoring", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,200,255), 2)
        
        # Status line
        status_text = f"HR: {last_heart_rate:.0f} BPM" if last_heart_rate else "HR: Measuring..."
        if last_gaze_data:
            status_text += " | Gaze: Active"
        cv2.putText(display, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)
        
        # Buffer status
        hr_buffer_progress = len(detector.hr_processor.frame_buffer) / 150
        hr_buffer_text = f"HR Buffer: {hr_buffer_progress:.0%}"
        cv2.putText(display, hr_buffer_text, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        # Mode info
        mode_text = f"Weights: Face {detector.face_weight:.0%} / Audio {detector.audio_weight:.0%}"
        cv2.putText(display, mode_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        
        # Feedback
        cv2.putText(display, "Integrated Care Guidance:", (10, H-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,200,255), 2)
        
        # Wrap feedback
        words = current_feedback.split()
        lines, line = [], []
        for w in words:
            if len(" ".join(line + [w])) < 70:
                line.append(w)
            else:
                lines.append(" ".join(line))
                line = [w]
        if line:
            lines.append(" ".join(line))
        
        for i, ln in enumerate(lines[:3]):
            y0 = H - 30 - i*20
            cv2.putText(display, ln, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.imshow("Multimodal Biometric Monitor with Gaze", display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            print("\n⚙️  Adjust weights")
            try:
                face_w = float(input("Face weight (0-1): "))
                audio_w = float(input("Audio weight (0-1): "))
                total = face_w + audio_w
                detector.face_weight = face_w / total
                detector.audio_weight = audio_w / total
                print(f"Updated: Face {detector.face_weight:.0%}, Audio {detector.audio_weight:.0%}")
            except:
                print("Invalid input")
        elif key == ord('h'):
            show_hr_details = not show_hr_details
            print(f"Heart rate display: {'ON' if show_hr_details else 'OFF'}")
        elif key == ord('g'):
            show_gaze_details = not show_gaze_details
            print(f"Gaze display: {'ON' if show_gaze_details else 'OFF'}")

finally:
    detector.stop_audio()
    cap.release()
    cv2.destroyAllWindows()
    finalize_feedback_file()
    
    # Print session summary
    print("\n📊 Session Summary:")
    if last_heart_rate:
        hr_data = list(detector.hr_processor.heart_rates)
        if hr_data:
            print(f"  Average HR: {np.mean(hr_data):.1f} BPM")
            print(f"  HR Range: {np.min(hr_data):.0f} - {np.max(hr_data):.0f} BPM")
    print("\nClosed successfully!")
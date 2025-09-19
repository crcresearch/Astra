import cv2
import torch
import tensorflow as tf
import onnx
import onnxruntime as ort


class MultimodalBiometricInferencer:
    """This class represents a multimodal biometric inferencer.

    :param emotion_model_path: Path to the emotion model, defaults to None
    :type emotion_model_path: str, optional
    :param audio_model_path: Path to the audio model, defaults to None
    :type audio_model_path: str, optional
    :param hr_model_path: Path to the HR model, defaults to None
    :type hr_model_path: str, optional
    :param gaze_model_path: Path to the gaze model, defaults to None
    :type gaze_model_path: str, optional
    :param patient_id: Patient ID, defaults to None
    :type patient_id: str, optional
    :param device: Device to use for inference, defaults to 'cuda'
    :type device: str, optional
    """
    def __init__(self, emotion_model_path=None, audio_model_path=None, hr_model_path=None, gaze_model_path=None, patient_id=None, device='cuda'):
        self.device = device
        self.patient_id = patient_id

        # Load the face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.emotion_model = self._load_tf_model(emotion_model_path) if emotion_model_path else None
        self.audio_model = self._load_tf_model(audio_model_path) if audio_model_path else None
        self.hr_model = self._load_tf_model(hr_model_path) if hr_model_path else None
        self.gaze_model = self._load_onnx_model(gaze_model_path) if gaze_model_path else None

    def _load_pytorch_model(self, model_path):
        """Load a PyTorch model from a TorchScript file.

        :param model_path: Path to the model file
        :type model_path: str
        :return: The loaded model
        :rtype: torch.nn.Module
        """
        model = torch.jit.load(model_path, map_location=self.device)
        model.to(self.device)
        if self.is_model_on_gpu(model):
            print(f"Model {model_path} is on GPU")
        else:
            print(f"Model {model_path} is on CPU")
        return model

    def _load_tf_model(self, model_path):
        """Load a TensorFlow model from a SavedModel file.

        :param model_path: Path to the model file
        :type model_path: str
        :return: The loaded model
        :rtype: tf.keras.Model
        """
        model = tf.keras.models.load_model(model_path)
        # compile the model?
        if self.is_model_on_gpu(model):
            print(f"Model {model_path} is on GPU")
        else:
            print(f"Model {model_path} is on CPU")
        return model

    def _load_onnx_model(self, model_path):
        raise NotImplementedError("ONNX model loading is not implemented")

    def predict_emotion(self, frame):
        # Takes a face
        raise NotImplementedError("Emotion prediction is not implemented")

    def predict_audio(self, frame):
        raise NotImplementedError("Audio prediction is not implemented")

    def predict_hr(self, frame):
        # Takes a face
        raise NotImplementedError("HR prediction is not implemented")

    def predict_gaze(self, frame):
        # Takes a face
        raise NotImplementedError("Gaze prediction is not implemented")

    def run_inference(self, frame):
        faces = self.face_cascade.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            self.predict_emotion(face)
            self.predict_hr(face)
            self.predict_gaze(face)
        raise NotImplementedError("Inference is not implemented")

    def is_model_on_gpu(self, model):
        if self.device != 'cuda':
            return False
        if isinstance(self.model, torch.nn.Module):
            return next(model.parameters()).is_cuda
        if isinstance(self.model, tf.keras.Model):
            # There doesn't seem to be a good way to check if a model is on GPU in TensorFlow, so just test if the GPU is found and hope for the best.
            return tf.test.is_gpu_available()
        # TODO: Add support for ONNX models
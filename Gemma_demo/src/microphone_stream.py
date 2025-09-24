import numpy as np
from stream import BaseStream

class MicrophoneStream(BaseStream):
    def __init__(self):
        # TODO: initialize mic stream here (PyAudio, sounddevice, etc.)
        pass

    def start(self):
        print("[MIC] MicrophoneStream start method not implemented")

    def stop(self):
        print("[MIC] MicrophoneStream stop method not implemented")

    def read(self):
        # TODO: return audio buffer
        print("[MIC] MicrophoneStream read method not implemented")
    
    def simulate_read(self):
        return np.random.uniform(-1.0, 1.0, (1024,)).astype(np.float32)
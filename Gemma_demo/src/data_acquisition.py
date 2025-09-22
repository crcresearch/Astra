import time
import random
import numpy as np
from camera_stream import CameraStream
from microphone_stream import MicrophoneStream
from sensor_array_stream import SensorArrayStream

class BaseDevice:
    """Abstract base class for any recording hardware."""
    def start(self):
        """Start the device."""
        raise NotImplementedError("Start method not implemented")

    def stop(self):
        """Stop the device."""
        raise NotImplementedError("Stop method not implemented")

    def read(self):
        """Return the latest data sample (frame, audio buffer, sensor reading, etc.)."""
        raise NotImplementedError("Read method not implemented")
    
    def simulate_read(self):
        """Return a simulated reading sample for testing without actual hardware."""
        raise NotImplementedError("Simulate read method not implemented")


class CameraDevice(BaseDevice):
    def __init__(self, camera_stream: CameraStream):
        self.camera = camera_stream

    def start(self):
        self.camera.start()

    def stop(self):
        self.camera.stop()

    def read(self):
        return self.camera.read()

    def simulate_read(self):
        return np.random.randint(0, 255, (640, 360, 3), dtype=np.uint8)


class MicrophoneDevice(BaseDevice):
    def __init__(self, microphone_stream: MicrophoneStream):
        # TODO: initialize mic stream here (PyAudio, sounddevice, etc.)
        self.microphone = microphone_stream

    def start(self):
        print("[MIC] MicrophoneDevice start method not implemented")

    def stop(self):
        print("[MIC] MicrophoneDevice stop method not implemented")

    def read(self):
        # TODO: return audio buffer
        print("[MIC] MicrophoneDevice read method not implemented")
    
    def simulate_read(self):
        return np.random.uniform(-1.0, 1.0, (1024,)).astype(np.float32)


class SensorDevice(BaseDevice):
    def __init__(self, sensor_stream: SensorArrayStream):
        self.sensor_stream = sensor_stream

    def start(self):
        print("[SENSOR] SensorDevice start method not implemented")

    def stop(self):
        print("[SENSOR] SensorDevice stop method not implemented")

    def read(self):
        print("[SENSOR] SensorDevice read method not implemented")

    def simulate_read(self):
        simulated_values = {
            "temperature": round(random.uniform(20, 35), 2),
            "humidity": random.randint(20, 80),
            "light": random.randint(0, 100),
            "gas": round(random.uniform(0, 1), 4),
        }
        return simulated_values


class DataAcquisition:
    """Orchestrator for all recording devices."""
    def __init__(self, streams=None, voice_activation=True):
        # Initialize devices from streams
        self.devices = []
        for stream in streams:
            if isinstance(stream, CameraStream):
                self.devices.append(CameraDevice(stream))
            elif isinstance(stream, MicrophoneStream):
                self.devices.append(MicrophoneDevice(stream))
            elif isinstance(stream, SensorArrayStream):
                self.devices.append(SensorDevice(stream))
            else:
                raise ValueError(f"Unsupported stream type: {type(stream)}")
        self.voice_activation = voice_activation
        self.is_recording = False

    def add_device(self, device: BaseDevice):
        self.devices.append(device)

    def start(self):
        if self.voice_activation:
            self._wait_for_voice_activation()
        else:
            self._start_all()

    def _wait_for_voice_activation(self):
        print("[DAQ] Waiting for voice activation...")
        # TODO: replace with actual voice activation trigger
        time.sleep(3)
        self._start_all()

    def _start_all(self):
        print("[DAQ] Starting all devices...")
        for device in self.devices:
            device.start()
        self.is_recording = True

    def stop(self):
        print("[DAQ] Stopping all devices...")
        for device in self.devices:
            device.stop()
        self.is_recording = False

    def collect_data(self):
        """Collect one sample from each device."""
        if not self.is_recording:
            return {}
        data = {}
        for device in self.devices:
            data[device.__class__.__name__] = device.read()
        return data

    def simulate_collect_data(self):
        """Simulate collecting one sample from each device.
        """
        data = {}
        for device in self.devices:
            data[device.__class__.__name__] = device.simulate_read()
        return data


# Example usage
if __name__ == "__main__":
    #pipeline = "rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2"
    pipeline = (
            "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 protocols=tcp latency=60 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! appsink sync=false max-buffers=1 drop=true"
            )
    
    camera_stream = CameraStream(pipeline, gstreamer=True)
    microphone_stream = MicrophoneStream()
    sensor_array_stream = SensorArrayStream()

    daq = DataAcquisition(streams=[camera_stream, microphone_stream, sensor_array_stream])

    daq.start()

    for _ in range(3):
        #sample = daq.collect_data()
        sample = daq.simulate_collect_data()
        print("Sample:", sample)
        time.sleep(1)

    daq.stop()
    

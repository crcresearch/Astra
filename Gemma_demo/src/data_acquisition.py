import time
import random
import numpy as np
import threading
import queue
from camera_stream import CameraStream
from microphone_stream import MicrophoneStream
from sensor_array_stream import SensorArrayStream
from stream import BaseStream

class DataAcquisition:
    """Orchestrator for recording streams for data collection.

    :param streams: List of streams to use. All streams must be instances of BaseStream.
    :type streams: list[BaseStream]
    :param voice_activation: Enable voice activation, defaults to True
    :type voice_activation: bool, optional
    :param auto_stop_after_seconds: Number of seconds to record after voice activation before stopping, defaults to None
    :type auto_stop_after_seconds: int, optional
    """
    def __init__(self, streams, voice_activation=True, auto_stop_after_seconds=None):
        # Check and filter out unsupported streams.
        self.device_streams = []
        for stream in streams:
            if not isinstance(stream, BaseStream):
                print(f"[DAQ][WARNING] Unsupported stream type: {type(stream)}\nRemoved from the stream list.")
            else:
                self.device_streams.append(stream)
        self.voice_activation = voice_activation
        self.is_recording = False
        # Voice listener internals
        self._voice_cmd_queue = queue.Queue()
        self._voice_listener_thread = None
        # Auto stop not implemented yet
        if auto_stop_after_seconds:
            raise NotImplementedError("Auto stop not implemented yet.\nPlease leave auto_stop_after_seconds as None.")
        self.auto_stop_after_seconds = auto_stop_after_seconds

    def add_device(self, stream: BaseStream):
        """Add a stream to the data acquisition pipeline.

        :param stream: The stream to add
        :type stream: BaseStream
        """
        self.device_streams.append(stream)
        print(f"[DAQ] Added stream: {stream.__class__.__name__}")

    def start(self):
        """Start the data acquisition pipeline. If voice activation is enabled, it will wait for voice activation. Otherwise, it will start all streams.
        """
        if self.voice_activation:
            self._wait_for_voice_activation()
        else:
            self._start_all()

    def _wait_for_voice_activation(self):
        """Ensure voice listener is started and wait for voice activation.
        """
        self._ensure_voice_listener_started()
        print("[DAQ] Voice control active. Say 'start' to begin, 'stop' to end.")

    def _start_all(self):
        """Start all streams.
        """
        if self.is_recording:
            print("[DAQ] Already recording. Please stop the current recording session before starting a new one.")
            return
        print("[DAQ] Starting all streams...")
        for device in self.device_streams:
            try:
                device.start()
                print(f"[DAQ] Started stream: {device.__class__.__name__}")
                # Only set is_recording to True if a stream is successfully started.
                self.is_recording = True
            except Exception as e:
                print(f"[DAQ] Failed to start stream: {device.__class__.__name__} - {e}")

    def stop(self):
        """Stop all streams.
        """
        if not self.is_recording:
            print("[DAQ] No recording session is active.")
            return
        print("[DAQ] Stopping all streams...")
        for device in self.device_streams:
            try:
                device.stop()
                print(f"[DAQ] Stopped stream: {device.__class__.__name__}")
            except Exception as e:
                print(f"[DAQ] Failed to stop stream: {device.__class__.__name__} - {e}")
                print("[DAQ][WARNING] Proper handling of failed stream stop is not implemented yet.")
        self.is_recording = False

    def collect_data(self):
        """Collect one sample from each device.

        :return: A dictionary of data from each device. If no recording session is active, returns None.
        :rtype: dict or None
        """
        if not self.is_recording:
            print("[DAQ] No recording session is active. Please start a recording session before collecting data.")
            return None
        data = {}
        for device in self.device_streams:
            try:
                data[device.__class__.__name__] = device.read()
            except Exception as e:
                print(f"[DAQ] Failed to collect data from {device.__class__.__name__} - {e}")
                print("[DAQ][WARNING] Proper handling of failed data collection is not implemented yet.")
        return data

    def simulate_collect_data(self):
        """Simulate collecting one sample from each device.

        :return: A dictionary of simulated data from each device.
        :rtype: dict
        """
        data = {}
        for device in self.device_streams:
            data[device.__class__.__name__] = device.simulate_read()
        return data

    def _ensure_voice_listener_started(self):
        """Start the voice listener thread if it is not already running.
        """
        if self._voice_listener_thread and self._voice_listener_thread.is_alive():
            return
        # Drain any stale commands before starting a new session.
        try:
            while True:
                self._voice_cmd_queue.get_nowait()
        except queue.Empty:
            pass
        self._voice_listener_thread = threading.Thread(
            target=self._run_voice_listener,
            name="DAQVoiceListener",
            daemon=True,
        )
        self._voice_listener_thread.start()

    def _run_voice_listener(self):
        """Worker thread for voice listener.
        """
        while True:
            try:
                command = self._voice_cmd_queue.get(timeout=0.5)
                print(f"[DAQ][DEBUG] Dequeued voice command: {command}")
            except Exception:
                continue
            if not isinstance(command, str):
                continue
            normalized = command.strip().lower()
            # Sentinel: always stop streams, then exit listener.
            if normalized == "":
                print("[DAQ] Voice listener received sentinel (empty) command. Stopping streams and exiting.")
                self.stop()
                break
            if normalized == "start":
                if not self.is_recording:
                    print("[DAQ][DEBUG] Voice listener received start command. Starting streams.")
                    self._start_all()
            elif normalized == "stop":
                if self.is_recording:
                    print("[DAQ][DEBUG] Voice listener received stop command. Stopping streams.")
                    self.stop()

    def simulate_voice_command(self, command: str):
        """Simulate a voice command.

        :param command: The voice command to simulate
        :type command: str
        """
        self._ensure_voice_listener_started()
        self._voice_cmd_queue.put(command)
        print(f"[DAQ][DEBUG] Enqueued voice command: {command}")

    def shutdown_voice_listener(self):
        """Shutdown the voice listener.
        """
        # Request shutdown via sentinel if a listener thread is alive.
        if self._voice_listener_thread and self._voice_listener_thread.is_alive():
            self._voice_cmd_queue.put("")
            self._voice_listener_thread.join(timeout=2.0)
            # Mark as no longer running
            if not self._voice_listener_thread.is_alive():
                self._voice_listener_thread = None


# Example usage
if __name__ == "__main__":
    pipeline = "rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2"
    #pipeline = (
    #        "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 protocols=tcp latency=60 ! "
    #        "rtph264depay ! h264parse ! nvv4l2decoder ! "
    #        "nvvidconv ! video/x-raw, format=BGRx ! "
    #        "videoconvert ! appsink sync=false max-buffers=1 drop=true"
    #        )
    
    #camera_stream = CameraStream(pipeline, gstreamer=False)
    microphone_stream = MicrophoneStream()
    sensor_array_stream = SensorArrayStream()

    daq = DataAcquisition(streams=[microphone_stream, sensor_array_stream])

    # Start listening for voice activation
    daq.start()
    time.sleep(1)
    # On voice activation, collect data
    daq.simulate_voice_command("start")

    for _ in range(3):
        #sample = daq.collect_data()
        sample = daq.simulate_collect_data()
        print("Sample:", sample)
        time.sleep(1)

    # Stop voice listener
    daq.simulate_voice_command("stop")
    daq.shutdown_voice_listener()

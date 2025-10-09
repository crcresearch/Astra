import cv2, threading, time
import numpy as np
from stream import BaseStream

class CameraStream(BaseStream):
    """Class to handle the camera stream to grab the latest frame from the camera stream.

    :param pipeline: The pipeline to use to open the camera stream.
    :type pipeline: str
    """
    def __init__(self, pipeline: str, gstreamer: bool = True):
        self._using_gstreamer = gstreamer
        if gstreamer:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(pipeline)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video stream with pipeline: {pipeline}")

        # Try to reduce internal buffering if backend supports it.
        # Note: GStreamer backend does not support CAP_PROP_BUFFERSIZE.
        if not self._using_gstreamer:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                print(f"Cannot set buffer size to 1: {e}")

        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self._first_frame_event = threading.Event()

    def start(self):
        """Start the camera stream.

        :return: The camera stream.
        :rtype: CameraStream
        """
        if self.running:
            return self
        self.running = True
        self._first_frame_event.clear()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        self._wait_for_first_frame()
        return self

    def _update(self):
        """Update the camera stream.
        """
        while self.running:
            # Grab before read to get a stamp as close as possible to the frame.
            # TODO: get PTS, check if not zero, convert to nano seconds.
            self.timestamp = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) * 1000000) if self.cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else 0
            ret, frame = self.cap.read()
            if not ret:
                # Avoid tight spin if camera hiccups
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
            if not self._first_frame_event.is_set():
                self._first_frame_event.set()

    def read(self):
        """Read the latest frame from the camera stream.

        :return: The latest frame from the camera stream with a timestamp or None if there is no frame.
        :rtype: tuple of numpy.ndarray and int or None
        """
        if not self.running:
            print("[ERROR] Camera stream is not running")
            return None
        with self.lock:
            return None if self.frame is None else (self.frame.copy(), self.timestamp)

    def _wait_for_first_frame(self, timeout: float = 2.0) -> bool:
        """Block until the first frame is available or timeout elapses.

        :param timeout: Seconds to wait at most.
        :return: True if first frame arrived, False on timeout.
        :rtype: bool
        """
        return self._first_frame_event.wait(timeout=timeout)

    def stop(self):
        """Stop the camera stream.
        """
        if not self.running:
            return
        self.running = False
        # Join thread to ensure clean shutdown
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self.cap.release()

    def simulate_read(self):
        return np.random.randint(0, 255, (640, 360, 3), dtype=np.uint8)

    # Context manager support
    def __enter__(self):
        """Upon entering the context manager, start the camera stream.
        """
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        """Upon exiting the context manager, stop the camera stream.
        """
        self.stop()
        # Do not suppress exceptions
        return False

# Test the camera stream
if __name__ == "__main__":
    #pipeline = "rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2"
    # We may need to add "rtspsrc do-rtsp-keep-alive=true" to the pipeline
    pipeline = (
            "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 protocols=tcp latency=60 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! appsink sync=false max-buffers=1 drop=true"
            )

    camera = CameraStream(pipeline, gstreamer=True)
    camera.start()
    while True:
        frame, timestamp = camera.read()
        if frame is not None:
            print(f"Timestamp: {timestamp}")
            cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()
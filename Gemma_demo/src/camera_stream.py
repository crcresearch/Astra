import cv2, threading, time

class CameraStream:
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
        # Note: GStreamer backend does not support CAP_PROP_BUFFERSIZE and may warn/fail.
        if not self._using_gstreamer:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                print(f"Cannot set buffer size to 1: {e}")

        self.frame = None
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
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
        return self

    def update(self):
        """Update the camera stream.
        """
        while self.running:
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

        :return: The latest frame from the camera stream or None if there is no frame.
        :rtype: numpy.ndarray or None
        """
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def wait_for_first_frame(self, timeout: float = 2.0) -> bool:
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
            self._thread.join(timeout=1.0)
        self._thread = None
        self.cap.release()

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
    #pipeline = "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 latency=0 ! nvgstreamer src=nvgstreamer ! video/x-raw(memory:NVMM), width=1920, height=1080, format=I420, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
    #pipeline = "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
    # Pipeline straight from Jetson forums: https://forums.developer.nvidia.com/t/doesnt-work-nvv4l2decoder-for-decoding-rtsp-in-gstreamer-opencv/140321
    #pipeline = "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink "
    # Recommended:
    pipeline = "rtspsrc location=rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2 latency=60 protocols=udp drop-on-latency=true ! rtph264depay ! h264parse config-interval=-1 ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
    #pipeline = "rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2"
    camera = CameraStream(pipeline, gstreamer=True)
    camera.start()
    camera.wait_for_first_frame()
    while True:
        frame = camera.read()
        if frame is not None:
            cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()
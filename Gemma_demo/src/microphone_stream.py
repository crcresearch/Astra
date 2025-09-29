import numpy as np
import threading
from stream import BaseStream

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except Exception as e:  # pragma: no cover
    gi = None
    Gst = None
    _gi_import_error = e
else:
    _gi_import_error = None


class MicrophoneStream(BaseStream):
    """Audio stream from a provided GStreamer pipeline ending in an appsink.

    Provide the full pipeline string. It must include an appsink named "appsink".
    Set desired output caps in the pipeline (e.g., F32LE) before the appsink, e.g.:
    ... ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,channels=1,rate=8000 ! appsink name=appsink sync=false max-buffers=10 drop=true

    :param pipeline: Full GStreamer pipeline string (must include appsink name=appsink)
    :type pipeline: str
    :param ring_buffer_seconds: Number of seconds of audio to keep in the ring buffer, defaults to 5
    :type ring_buffer_seconds: int, optional
    :param channels: Number of channels in the audio stream. If left as None, an attempt will be made to determine the channel count from the pipeline, defaults to None
    :type channels: int, optional
    :param read_timeout_s: Timeout in seconds for read() when pulling a sample, defaults to 0.2
    :type read_timeout_s: float, optional
    :param sample_rate: Sample rate to use if it cannot be determined from the pipeline, defaults to None
    :type sample_rate: int, optional
    """

    def __init__(self, pipeline: str, ring_buffer_seconds=5, read_timeout_s: float = 0.2, channels: int | None = None, sample_rate: int | None = None):
        if _gi_import_error is not None:
            raise RuntimeError(
                f"Failed to import GStreamer (gi): {_gi_import_error}"
            )

        if not Gst.is_initialized():
            Gst.init(None)

        self._timeout_s = read_timeout_s
        self._channels = channels
        self._buffer_seconds = float(ring_buffer_seconds)
        self._sample_rate = sample_rate
        # Parse provided pipeline and locate the appsink named 'appsink'
        self.pipeline = Gst.parse_launch(pipeline)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("Pipeline must contain an appsink named 'appsink'")

        self.running = False
        self._latest = None
        self._lock = None
        self._thread = None
        # Ring buffer storage
        self._rbuff = []            # list of numpy arrays (frames[, channels])
        self._rbuff_frames = 0      # total frames stored
        self._rbuff_cap = 0         # capacity in frames

    def start(self):
        """Start the microphone stream.

        :return: The microphone stream.
        :rtype: MicrophoneStream
        """
        if self.running:
            return self
        state_ret = self.pipeline.set_state(Gst.State.PLAYING)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING state")
        # Keep a bus reference for non-blocking error polling
        self._bus = self.pipeline.get_bus()
        # Wait briefly for the pipeline to preroll/transition
        change_ret, current, pending = self.pipeline.get_state(timeout=Gst.SECOND * 2)
        if change_ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline failed to reach a valid state")
        # Quick non-blocking error check
        self._poll_bus_errors(non_blocking=True)
        self.running = True
        # Determine channel count from negotiated caps if not provided
        if self._channels is None:
            sink_pad = self.appsink.get_static_pad("sink")
            if sink_pad is not None:
                caps = sink_pad.get_current_caps()
                if caps is not None and caps.get_size() > 0:
                    structure = caps.get_structure(0)
                    if structure is not None and structure.has_field("channels"):
                        try:
                            self._channels = int(structure.get_value("channels"))
                            print(f"[MIC] Channel count determined from pipeline: {self._channels}")
                        except Exception:
                            self._channels = None
            if self._channels is None:
                print("[MIC][WARNING] Could not determine channel count from pipeline, defaulting to 1")
                self._channels = 1
        # Determine sample rate
        if self._sample_rate is None:
            sink_pad = self.appsink.get_static_pad("sink")
            if sink_pad is not None:
                caps = sink_pad.get_current_caps()
                if caps is not None and caps.get_size() > 0:
                    structure = caps.get_structure(0)
                    if structure is not None and structure.has_field("rate"):
                        try:
                            self._sample_rate = int(structure.get_value("rate"))
                            print(f"[MIC] Sample rate determined from pipeline: {self._sample_rate}")
                        except Exception:
                            self._sample_rate = None
        if self._sample_rate is None:
            # Conservative fallback
            self._sample_rate = 8000
            print("[MIC][WARNING] Could not determine sample rate from pipeline, defaulting to 8000")
        # Compute ring buffer capacity in frames
        self._rbuff_cap = int(self._buffer_seconds * self._sample_rate)
        # Initialize lock and start background puller
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the microphone stream.
        """
        if not self.running:
            return
        self.running = False
        # Join reader thread
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.pipeline.set_state(Gst.State.NULL)
        # Drain any remaining error messages (non-blocking)
        self._poll_bus_errors(non_blocking=True)
        print("[MIC][DEBUG] Microphone stream stopped")

    def read(self):
        """Return the most recent ring-buffer window.

        :return: Concatenated audio window or None if unavailable.
        :rtype: numpy.ndarray or None
        """
        if not self.running:
            print("[MIC] Microphone stream is not running")
            return None
        # Gather the last _buffer_seconds worth of frames
        with self._lock:
            if self._rbuff_frames <= 0:
                return None
            need = self._rbuff_cap if self._rbuff_cap > 0 else self._rbuff_frames
            need = min(need, self._rbuff_frames)
            chunks = []
            remaining = need
            for arr in reversed(self._rbuff):
                n = arr.shape[0] if arr.ndim > 1 else arr.size
                if remaining <= 0:
                    break
                if n >= remaining:
                    chunks.append(arr[-remaining:])
                    remaining = 0
                else:
                    chunks.append(arr)
                    remaining -= n
            if not chunks:
                return None
            window = np.concatenate(chunks[::-1], axis=0)
            return window.astype(np.float32, copy=False)

    def _update(self):
        """Background puller that keeps the latest audio sample fresh."""
        while self.running:
            # Check for errors or EOS to keep bus drained
            self._poll_bus_errors(non_blocking=True)
            timeout_ns = int(self._timeout_s * Gst.SECOND)
            sample = self.appsink.emit("try-pull-sample", timeout_ns)
            if sample is None:
                continue
            buffer = sample.get_buffer()
            if buffer is None:
                continue
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue
            try:
                arr = np.frombuffer(map_info.data, dtype=np.float32)
                if self._channels is not None and self._channels > 1:
                    frames = arr.size // self._channels
                    arr = arr[: frames * self._channels].reshape(frames, self._channels)
                with self._lock:
                    # Append to ring buffer
                    self._rbuff.append(arr.copy())
                    frames = arr.shape[0] if arr.ndim > 1 else arr.size
                    self._rbuff_frames += frames
                    # Evict old data beyond capacity
                    while self._rbuff_cap > 0 and self._rbuff_frames > self._rbuff_cap and self._rbuff:
                        oldest = self._rbuff[0]
                        drop = oldest.shape[0] if oldest.ndim > 1 else oldest.size
                        self._rbuff_frames -= drop
                        self._rbuff.pop(0)
            finally:
                buffer.unmap(map_info)
    
    def simulate_read(self):
        """Simulate reading a sample from the microphone stream.

        :return: A simulated audio sample.
        :rtype: numpy.ndarray
        """
        return np.random.uniform(-1.0, 1.0, (1024,)).astype(np.float32)

    def _poll_bus_errors(self, non_blocking: bool = True):
        """Poll the message bus for errors keeping the queue empty.

        :param non_blocking: Whether to poll the bus non-blocking, defaults to True
        :type non_blocking: bool, optional
        """
        if not hasattr(self, "_bus") or self._bus is None:
            return
        flags = Gst.MessageType.ERROR | Gst.MessageType.EOS
        # 0 timeout for non-blocking, small timeout otherwise
        timeout = 0 if non_blocking else int(0.1 * Gst.SECOND)
        while True:
            msg = self._bus.timed_pop_filtered(timeout, flags)
            if msg is None:
                break
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"[MIC][GStreamer ERROR] {err}; debug={debug}")
                self.running = False
                self.pipeline.set_state(Gst.State.NULL)
                break
            elif msg.type == Gst.MessageType.EOS:
                print("[MIC] End of stream received")
                self.running = False
                self.pipeline.set_state(Gst.State.NULL)
                break
    
    # Context manager support
    def __enter__(self):
        """Upon entering the context manager, start the microphone stream.
        """
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        """Upon exiting the context manager, stop the microphone stream.
        """
        self.stop()
        return False

if __name__ == "__main__":
    import time
    rtsp_uri = "rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2"
    pipeline = (
        f"rtspsrc location={rtsp_uri} latency=60 do-rtsp-keep-alive=true ! "
        "application/x-rtp,media=audio,encoding-name=PCMA,clock-rate=8000 ! "
        "rtppcmadepay ! alawdec ! audioconvert ! audioresample ! "
        "audio/x-raw,format=F32LE,channels=1,rate=8000 ! "
        "appsink name=appsink sync=false max-buffers=10 drop=true"
    )
    mic = MicrophoneStream(pipeline, read_timeout_s=2.0, channels=1)
    print(f"Starting stream")
    mic.start()
    print("Waiting for stream to stabilize...")
    for i in range(10):
        time.sleep(1)
        print(f"{10-i}")
    print("Make some noise!")
    for i in range(10):
        print(f"Attempt {i+1}/10")
        audio_data = mic.read()
        if audio_data is not None:
            print(f"Audio data shape: {audio_data.shape}")
            print(f"Audio data type: {audio_data.dtype}")
            print(f"Audio data range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            break
        time.sleep(1)
    else:
        print("No audio data received after 10 attempts")
    mic.stop()
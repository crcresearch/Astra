import numpy as np
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
    """RTSP audio stream using GStreamer appsink. This builds a pipeline roughly equivalent to:
    uridecodebin uri=rtsp://... ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,channels=<C>,rate=<R> ! appsink

    :param uri: RTSP URI to the source with audio
    :type uri: str
    :param sample_rate: Output sample rate (e.g., 16000 or 48000)
    :type sample_rate: int
    :param channels: Output number of channels (1=mono, 2=stereo)
    :type channels: int
    :param sample_format: Output sample format (default F32LE)
    :type sample_format: str
    :param read_timeout_s: Timeout in seconds for read() when pulling a sample
    :type read_timeout_s: float
    """

    def __init__(self, uri: str, sample_rate: int = 16000, channels: int = 1,
                 sample_format: str = "F32LE", read_timeout_s: float = 0.2):
        if _gi_import_error is not None:
            raise RuntimeError(
                f"Failed to import GStreamer (gi): {_gi_import_error}"
            )

        if not Gst.is_initialized():
            Gst.init(None)

        self.uri = uri
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_format = str(sample_format)
        self._timeout_s = float(read_timeout_s)

        # Build pipeline elements
        self.pipeline = Gst.Pipeline.new("audio-pipeline")

        self.decodebin = Gst.ElementFactory.make("uridecodebin", "src")
        if self.decodebin is None:
            raise RuntimeError("Failed to create uridecodebin")
        self.decodebin.set_property("uri", self.uri)

        self.convert = Gst.ElementFactory.make("audioconvert", "convert")
        if self.convert is None:
            raise RuntimeError("Failed to create audioconvert")

        self.resample = Gst.ElementFactory.make("audioresample", "resample")
        if self.resample is None:
            raise RuntimeError("Failed to create audioresample")

        self.appsink = Gst.ElementFactory.make("appsink", "sink")
        if self.appsink is None:
            raise RuntimeError("Failed to create appsink")

        # Configure appsink: request format and behavior
        caps_str = (
            f"audio/x-raw,format={self.sample_format},layout=interleaved,"
            f"channels={self.channels},rate={self.sample_rate}"
        )
        caps = Gst.Caps.from_string(caps_str)
        self.appsink.set_property("caps", caps)
        self.appsink.set_property("emit-signals", False)  # we'll poll
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 10)
        self.appsink.set_property("drop", True)

        # Assemble pipeline
        for elem in (self.decodebin, self.convert, self.resample, self.appsink):
            self.pipeline.add(elem)

        if not self.convert.link(self.resample):
            raise RuntimeError("Failed to link audioconvert -> audioresample")
        if not self.resample.link(self.appsink):
            raise RuntimeError("Failed to link audioresample -> appsink")

        # Deferred link from decodebin via pad-added
        self.decodebin.connect("pad-added", self._on_decodebin_pad_added)

        self.running = False

    def _on_decodebin_pad_added(self, decodebin, pad):
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        media_type = structure.get_name() if structure is not None else ""
        if not media_type.startswith("audio/"):
            return
        sink_pad = self.convert.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    def start(self):
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
        return self

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        # Drain any remaining error messages (non-blocking)
        self._poll_bus_errors(non_blocking=True)

    def read(self):
        if not self.running:
            print("[MIC] Microphone stream is not running")
            return None
        # Check for errors or EOS before pulling a sample
        self._poll_bus_errors(non_blocking=True)
        timeout_ns = int(self._timeout_s * Gst.SECOND)
        sample = self.appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return None
        buffer = sample.get_buffer()
        if buffer is None:
            return None
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            # Convert to numpy array. We requested F32LE interleaved
            audio_np = np.frombuffer(map_info.data, dtype=np.float32)
            if self.channels > 1:
                # Reshape as (num_frames, channels)
                frames = audio_np.size // self.channels
                audio_np = audio_np[: frames * self.channels].reshape(frames, self.channels)
            return audio_np.copy()
        finally:
            buffer.unmap(map_info)

    def simulate_read(self):
        return np.random.uniform(-1.0, 1.0, (1024,)).astype(np.float32)

    # Context manager support for parity with CameraStream
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # Internal helpers
    def _poll_bus_errors(self, non_blocking: bool = True):
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

if __name__ == "__main__":
    rtsp_uri = "rtsp://voice4pimd:voice4pimd@10.12.130.50/stream2"
    mic = MicrophoneStream(rtsp_uri)
    mic.start()
    # Grab single sample
    audio_data = mic.read()
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Audio data type: {audio_data.dtype}")
    print(f"Audio data: {audio_data}")
    mic.stop()
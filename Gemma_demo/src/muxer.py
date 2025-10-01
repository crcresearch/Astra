import numpy as np

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


class GStreamerMuxer:
    """GStreamer appsrc-based audio/video muxer.
    Accepts overlaid video frames (NumPy BGR) and
    audio chunks (NumPy float32, mono or interleaved) with caller-managed PTS and
    pushes them into a user-provided GStreamer pipeline string featuring two
    `appsrc` elements (video and audio), an `mp4mux`, and a `filesink`.
    Enables fully in-process muxing/encoding using GStreamer while allowing
    Python-side preprocessing (e.g., ML overlays) and precise timestamp control.

    Pipeline requirements (provided as a single string):
    - Must include two `appsrc` elements named `vid_src` and `aud_src`.
    - Must include an `mp4mux` named `mux` and a `filesink` named `out`.
    - The muxer sets the filesink `location` at runtime and applies appsrc caps
      to match provided video size/fps and audio rate/channels.

    Example (x264enc + AAC):
    ``appsrc name=vid_src is-live=true format=time do-timestamp=false ! videoconvert !
    x264enc speed-preset=ultrafast tune=zerolatency key-int-max=40 bitrate=4000000 !
    h264parse config-interval=-1 ! queue ! mux. appsrc name=aud_src is-live=true
    format=time do-timestamp=false ! audioconvert ! audioresample ! voaacenc
    bitrate=128000 ! aacparse ! queue ! mux. mp4mux name=mux faststart=true !
    filesink name=out location=output.mp4``

    :param pipeline_str: Full GStreamer pipeline string with `vid_src`, `aud_src`, `mux`, and `out` elements.
    :type pipeline_str: str
    :param output_path: Output file path for the `filesink` (e.g., "output.mp4").
    :type output_path: str
    :param video_width: Video frame width in pixels.
    :type video_width: int
    :param video_height: Video frame height in pixels.
    :type video_height: int
    :param video_fps_num: Video frames-per-second numerator (e.g., 20 for 20/1).
    :type video_fps_num: int
    :param video_fps_den: Video frames-per-second denominator (e.g., 1 for 20/1).
    :type video_fps_den: int
    :param audio_rate: Audio sample rate in Hz (e.g., 8000 or 16000).
    :type audio_rate: int
    :param audio_channels: Audio channel count (1=mono, 2=stereo).
    :type audio_channels: int
    :param video_format: Appsrc raw video format (e.g., "BGR" to match OpenCV frames), defaults to "BGR".
    :type video_format: str, optional
    :param audio_format: Appsrc raw audio format (e.g., "F32LE"), defaults to "F32LE".
    :type audio_format: str, optional
    :raises RuntimeError: If GStreamer is unavailable, pipeline parsing fails, or required elements are missing.
    """

    def __init__(
        self,
        pipeline_str: str,
        output_path: str,
        video_width: int,
        video_height: int,
        video_fps_num: int,
        video_fps_den: int,
        audio_rate: int,
        audio_channels: int,
        video_format: str = "BGR",
        audio_format: str = "F32LE",
    ) -> None:
        if _gi_import_error is not None:
            raise RuntimeError(f"Failed to import GStreamer (gi): {_gi_import_error}")
        if not Gst.is_initialized():
            Gst.init(None)

        self._video_width = video_width
        self._video_height = video_height
        self._video_fps_num = video_fps_num
        self._video_fps_den = video_fps_den
        self._video_format = video_format
        self._audio_rate = audio_rate
        self._audio_channels = audio_channels
        self._audio_format = audio_format

        self._pipeline = Gst.parse_launch(pipeline_str)
        if self._pipeline is None:
            raise RuntimeError("Failed to parse GStreamer pipeline string")

        self._vid_src = self._pipeline.get_by_name("vid_src")
        self._aud_src = self._pipeline.get_by_name("aud_src")
        self._mux = self._pipeline.get_by_name("mux")
        self._filesink = self._pipeline.get_by_name("out")

        if any(x is None for x in [self._vid_src, self._aud_src, self._mux, self._filesink]):
            raise RuntimeError("Pipeline must expose 'vid_src', 'aud_src', 'mux', and 'out' by name")

        # Route output path
        self._filesink.set_property("location", output_path)

        # Set appsrc caps to match provided dimensions/rates
        v_caps_str = (
            f"video/x-raw,format={self._video_format},width={self._video_width},"
            f"height={self._video_height},framerate={self._video_fps_num}/{self._video_fps_den}"
        )
        a_caps_str = (
            f"audio/x-raw,format={self._audio_format},layout=interleaved,"
            f"channels={self._audio_channels},rate={self._audio_rate}"
        )
        self._vid_src.set_property("caps", Gst.Caps.from_string(v_caps_str))
        self._aud_src.set_property("caps", Gst.Caps.from_string(a_caps_str))

        # Ensure time-based PTS handling
        self._vid_src.set_property("format", Gst.Format.TIME)
        self._aud_src.set_property("format", Gst.Format.TIME)
        # Appsrc timestamping is driven by caller PTS (do-timestamp=false)

        self._started = False
        self._video_pts_ns = 0
        self._frame_duration_ns = int(Gst.SECOND * self._video_fps_den // max(self._video_fps_num, 1))

    def start(self) -> None:
        """Start the muxer pipeline.

        Transitions the pipeline to PLAYING state so subsequent `push_video` and
        `push_audio` calls can feed data.

        :raises RuntimeError: If the pipeline cannot enter the PLAYING state.
        :return: None
        :rtype: None
        """
        if self._started:
            return
        change = self._pipeline.set_state(Gst.State.PLAYING)
        if change == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Failed to set muxer pipeline to PLAYING")
        self._started = True

    def stop(self) -> None:
        """Stop the muxer pipeline and finalize the output file.

        Sends EOS to both appsrc elements, transitions the pipeline to NULL, and
        releases internal state.

        :return: None
        :rtype: None
        """
        if not self._started:
            return
        try:
            self._vid_src.emit("end-of-stream")
            self._aud_src.emit("end-of-stream")
        except Exception:
            pass
        self._pipeline.set_state(Gst.State.NULL)
        self._started = False

    def push_video(self, frame_bgr: np.ndarray, pts_ns: int | None = None, duration_ns: int | None = None) -> None:
        """Push a single video frame into the muxer.

        :param frame_bgr: Video frame in BGR format (NumPy array of shape (H, W, 3)).
        :type frame_bgr: numpy.ndarray
        :param pts_ns: Presentation timestamp in nanoseconds. If None, an internal clock is used, defaults to None.
        :type pts_ns: int, optional
        :param duration_ns: Frame duration in nanoseconds. If None, derived from fps, defaults to None.
        :type duration_ns: int, optional
        :raises ValueError: If the frame size does not match the configured width/height.
        :return: None
        :rtype: None
        """
        if not self._started:
            return
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        if h != self._video_height or w != self._video_width:
            raise ValueError(f"Frame size {w}x{h} does not match configured {self._video_width}x{self._video_height}")
        data = memoryview(frame_bgr).tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = self._video_pts_ns if pts_ns is None else int(pts_ns)
        buf.duration = self._frame_duration_ns if duration_ns is None else int(duration_ns)
        if pts_ns is None:
            self._video_pts_ns += buf.duration
        self._vid_src.emit("push-buffer", buf)

    def push_audio(self, audio_chunk: np.ndarray, pts_ns: int, duration_ns: int | None = None) -> None:
        """Push an audio chunk into the muxer.

        The audio chunk should be float32 NumPy data, either mono (shape (N,)) or
        interleaved multi-channel (shape (N, C)). The duration is derived from
        `N / sample_rate` if not explicitly provided.

        :param audio_chunk: Audio samples (float32 mono or interleaved float32 (N, C)).
        :type audio_chunk: numpy.ndarray
        :param pts_ns: Presentation timestamp in nanoseconds for the first sample.
        :type pts_ns: int
        :param duration_ns: Duration in nanoseconds. If None, derived from chunk length, defaults to None.
        :type duration_ns: int, optional
        :return: None
        :rtype: None
        """
        if not self._started:
            return
        if audio_chunk is None:
            return
        # Expect float32 mono or float32 interleaved (frames, channels)
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32, copy=False)
        frames = audio_chunk.shape[0] if audio_chunk.ndim > 1 else audio_chunk.size
        data = memoryview(audio_chunk).tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = int(pts_ns)
        buf.duration = int(frames * Gst.SECOND // self._audio_rate) if duration_ns is None else int(duration_ns)
        self._aud_src.emit("push-buffer", buf)



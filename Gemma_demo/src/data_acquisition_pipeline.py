from data_acquisition import DataAcquisition
from camera_stream import CameraStream
from microphone_stream import MicrophoneStream
from sensor_array_stream import SensorArrayStream
from muxer import GStreamerMuxer

import yaml
import time

from gi.repository import Gst

Gst.init(None)

# Path to devices' config file
DEVICE_CONFIG_PATH = 'device_config.yaml'
OUTPUT_FILE = ""

sensor_data = None

# Load in config
with open(DEVICE_CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

avs = GStreamerMuxer(
    pipeline_str=config['audio_video_pipeline']['appsrc_av_pipeline'],
    output_path="output.mp4",
    video_width=640,
    video_height=360,
    video_fps_num=20, video_fps_den=1,
    audio_rate=8000, audio_channels=1,
    video_format="BGR", audio_format="F32LE"
)
# Start up GStreamer output pipeline.
avs.start()

# Build the camera pipeline
camera_pipeline = f"rtspsrc location=rtsp://{config['camera']['username']}:{config['camera']['password']}@{config['camera']['ip']}/{config['camera']['stream']} {config['camera']['gstreamer_pipeline']}"
# If testing on non-gstreamer camera, use the following pipeline with gstreamer=False:
#camera_pipeline = f"rtsp://{config['camera']['username']}:{config['camera']['password']}@{config['camera']['ip']}/{config['camera']['stream']}"

# Build the microphone pipeline from YAML gstreamer_pipeline
mic_pipeline = (
    f"rtspsrc location=rtsp://{config['camera']['username']}:{config['camera']['password']}@{config['camera']['ip']}/{config['camera']['stream']} "
    f"{config['microphone']['gstreamer_pipeline']}"
)

# Load in hardware streamers
camera_stream = CameraStream(camera_pipeline, gstreamer=config['camera']['gstreamer'])
microphone_stream = MicrophoneStream(mic_pipeline, channels=config['microphone']['channels'])
sensor_array_stream = SensorArrayStream()

# Load in data acquisition pipeline
daq = DataAcquisition(streams=[camera_stream, microphone_stream, sensor_array_stream], voice_activation=True,
                      video_fps_num=config['camera']['video_fps_numerator'], video_fps_den=config['camera']['video_fps_denominator'],
                      audio_sample_rate=config['microphone']['audio_sample_rate'], audio_channels=config['microphone']['audio_channels'])

# Start data acquisition pipeline
daq.start() # If voice activation is enabled, it will wait for voice activation. Otherwise, it will start all streams.

# Simulate voice activation
daq.simulate_voice_command("start")

# Allow time for the streams to start
time.sleep(5)

# Collect data at 20Hz
target_sample_rate = 20
while True:
    try:
        sample = daq.collect_data()
        #print("Sample:", sample)
        # Push audio and video frames to muxer
        if sample and "video_frame" in sample and "audio_frame" in sample:
            avs.push_video(sample['video_frame'], sample['video_pts_ns'])
            avs.push_audio(sample['audio_frame'], sample['audio_pts_ns'])
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        # Stop the muxer GStreamer and therefore save to an output file.
        avs.stop()
        break
    except Exception as e:
        print(f"Error: {e}")
        break
    time.sleep(1/target_sample_rate)

# Simulate voice command to stop
daq.simulate_voice_command("stop")

# Shutdown voice listener
daq.shutdown_voice_listener()

# Safety stop to ensure the muxer is stopped incase there's an error in the main loop.
avs.stop()

# If you are not using the voice listener, you can stop the data acquisition pipeline manually with:
#daq.stop()
from data_acquisition import DataAcquisition
from camera_stream import CameraStream
from microphone_stream import MicrophoneStream
from sensor_array_stream import SensorArrayStream

import yaml
import time

# Path to devices' config file
DEVICE_CONFIG_PATH = 'device_config.yaml'

# Load in config
with open(DEVICE_CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Build the camera pipeline
camera_pipeline = f"rtspsrc location=rtsp://{config['camera']['username']}:{config['camera']['password']}@{config['camera']['ip']}/{config['camera']['stream']} {config['camera']['gstreamer_pipeline']}"
# If testing on non-gstreamer camera, use the following pipeline with gstreamer=False:
#camera_pipeline = f"rtsp://{config['camera']['username']}:{config['camera']['password']}@{config['camera']['ip']}/{config['camera']['stream']}"

# Load in hardware streamers
camera_stream = CameraStream(camera_pipeline, gstreamer=config['camera']['gstreamer'])
microphone_stream = MicrophoneStream()
sensor_array_stream = SensorArrayStream()

# Load in data acquisition pipeline
daq = DataAcquisition(streams=[camera_stream, microphone_stream, sensor_array_stream])

# Start data acquisition pipeline
daq.start()

# Collect data
for _ in range(3):
    #sample = daq.collect_data()
    sample = daq.simulate_collect_data()
    print("Sample:", sample)
    time.sleep(1)

# Stop data acquisition pipeline
daq.stop()
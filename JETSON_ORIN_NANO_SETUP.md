# Jetson Orin Nano Setup Instructions
## Hardware Setup Procedure
### Image Flashing Procedure
Mainly following the instructions here: [Jetson AI Lab initial setup](https://www.jetson-ai-lab.com/initial_setup_jon.html) and [NVIDIA Jetson Orin Nano getting started](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#write).
Steps:
1. Formatting the MicroSD
- Connect MicroSD to computer
- Format the MicroSD: download, install, run
- https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/
- Select the MicroSD card. Use the quick format option. Leave volume label blank. Format.
> ⚠️ Formatting the MicroSD will wipe it in the process.
2. Flashing the MicroSD with JetPack 6.2.1
> Our Jetson Orin Nanos already have the updated firmware (36.+); therefore we do not need to update the firmware.
- Download the JetPack 6.2.1 image: [JetPack 6.2.1 SDK](https://developer.nvidia.com/embedded/jetpack-sdk-621)
- Choose the "For Jetson Orin Nano Developer Kit currently running JetPack 6.x" option and download the MicroSD card image
> As we have the updated firmware, we can use this image right off the bat.
- Download, install, and run balenaEtcher: [balenaEtcher](https://etcher.balena.io/)
- Choose "Flash from file" and select the downloaded JetPack 6.2.1 zip file
- Select the MicroSD as the target and then hit "Flash!"
> This will take some time — 30 minutes at least, likely more.
3. Booting Up the Jetson with the Flashed Image
- Insert the MicroSD card into the Jetson Orin Nano's MicroSD card slot (under the board with the fan, on the outside edge)
- Connect a mouse and keyboard + monitor
- Plug in the power cable
- Should boot up with the image!  

Notes:  
- You will likely have to do a couple of reboot cycles, especially once you connect to the internet as the automatic updates install.
If boot fails:
- Use the boot menu to select the MicroSD as the default boot device.
- From the boot menu, you can also confirm the firmware version is 36.+ if needed.
### Camera
To set up the Tapo C120 for use with the Jetson Orin Nano, follow the instructions in the box and install the TP-Link Tapo app.  
Create a Tapo account and follow the on-screen steps to add the camera.  
After the camera is added, open it in the app, tap the settings icon (top-right), then go to "Advanced Settings" → "Camera Account" and create a camera account (this is different from your app account).  
Find your camera's IP address in "Advanced Settings" → "Network Settings". While you're there, turn on "Static IP".
#### Test Script
Simple Python OpenCV script (run this after the [Python Virtual Environment Setup](#python-virtual-environment-setup))
```python
import cv2

# Camera account credentials (set these to your camera account)
user = "<camera-account-username>"
password = "<camera-account-password>"
camera_ip = "<camera-ip-address>"

# Choose stream: /stream1 (higher quality) or /stream2 (lower latency/bitrate)
rtsp_url = f"rtsp://{user}:{password}@{camera_ip}/stream2"
print(f"Attempting to reach {rtsp_url}")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Unable access the camera")
    print("Check crendentials/IP and confirm the Jetson is on the same network as the camera.")
else:
    print("Camera accessed")

while cap.isOpened:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Captured Frame", frame)
    else:
        print("Error: Could not capture frame.")
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```
Ensure both the camera and the Jetson Orin Nano are on the same network; otherwise the Jetson will not be able to access the camera.
### Adafruit Sensor
We have a sensor for air temperature, humidity, and light level. TBD setup instructions.
### Quality of Life Things
#### Firefox installation
The Snap package may not work properly on JetPack 6.x. Use Mozilla's PPA instead (friends don't let friends use Chrome).
Remove the Snap installation and purge leftovers:
```bash
sudo snap remove firefox
sudo apt purge firefox
```
Add the Mozilla Team PPA:
```bash
sudo add-apt-repository ppa:mozillateam/ppa
sudo apt update
```
Pin the PPA so that APT will use it over the Snap:
```bash
echo '
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
' | sudo tee /etc/apt/preferences.d/mozilla-firefox
```
Install Firefox:
```bash
sudo apt install firefox
```
This pulls the .deb directly from Mozilla's PPA instead of the broken/incompatible Snap.
#### Jetson Stats (jtop) Installation
[Jetson Stats](https://rnext.it/jetson_stats/) is a useful utility that shows system performance information such as GPU utilization.
To install it, run (outside of your virtual environment. Type `deactivate` to deactivate it):
```bash
sudo pip3 install -U jetson-stats
```
> You may need to install pip for the system as JP6.2.1 comes with Python installed, but not pip:
```bash
sudo apt-get install python-pip python3-pip
```
Once installed, simply run:
```bash
jtop
```
### Swap Memory Increase Enable and Disable Shell Scripts
The Jetson Orin Nano only has 8 GB of RAM. This is commonly not enough to run the LMs we wish to run or the work we wish to do.
To "solve" that we set up more swap memory. Swap memory uses storage as memory when needed. As you can imagine, this is much slower than RAM, but still useful.  

First, create these scripts:
swap12g-on.sh
```bash
#!/bin/bash
set -e

echo "Disabling any active swap..."
sudo swapoff -a || true

echo "Creating 12G swapfile..."
sudo fallocate -l 12G /swapfile

echo "Setting permissions..."
sudo chmod 600 /swapfile

echo "Formatting swapfile..."
sudo mkswap /swapfile

echo "Enabling swap..."
sudo swapon /swapfile

echo "✅ Swap enabled:"
swapon --show
free -h
```
swap12g-off.sh
```bash
#!/bin/bash
set -e

if [ -f /swapfile ]; then
    echo "Disabling swap..."
    sudo swapoff /swapfile

    echo "Removing swapfile..."
    sudo rm /swapfile

    echo "✅ Swap removed."
else
    echo "No /swapfile found — nothing to clean up."
fi

echo
swapon --show
free -h
```
swap4g-on.sh
```bash
#!/bin/bash
set -e

echo "Disabling any active swap..."
sudo swapoff -a || true

echo "Creating 4G swapfile..."
sudo fallocate -l 4G /swapfile

echo "Setting permissions..."
sudo chmod 600 /swapfile

echo "Formatting swapfile..."
sudo mkswap /swapfile

echo "Enabling swap..."
sudo swapon /swapfile

echo "✅ Swap enabled:"
swapon --show
free -h
```
Make them executable:
```bash
chmod +x swap12g-on.sh swap12g-off.sh swap4g-on.sh
```
Enable 12G swap memory:
```bash
./swap12g-on.sh
```
Disable swap memory:
```bash
./swap12g-off.sh
```
Restore default 4G of swap memory
```bash
swap4g-on.sh
```
The swap memory is 4G by default and the change to the swap memory is not permanent. A reboot will reset the swap memory configuration.
### Ollama Installation
Simply run the installer:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
#### Pull and Run gemma3n:e2b
```bash
# Pull
ollama pull gemma3n:e2b

# Run
ollama run gemma3n:e2b
```
> ⚠️ Ensure you have increased swap memory before running the gemma3n:e2b model.
## Python Virtual Environment Setup
Create a Python virtual environment:
```bash
python3 -m venv <venv-name-here>
```
Activate it:
```bash
source <venv-name-here>/bin/activate
```
### PyTorch (+ torchvision)
Thankfully, there are prebuilt torch and torchvision wheels with CUDA for the Jetson at [Jetson AI Lab - cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126)  
We will be using these.
1. Install needed system packages
- libopenblas-dev
```bash
sudo apt-get install -y libopenblas-dev
```
- cuSPARSELt  
Thankfully, NVIDIA provides an installer for this:
```bash
# Download the installer.
wget https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

# Run the installer and then run the command it spits out.
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

# This command is a template of what the above command spits out.
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/

# Update
sudo apt-get update

# And install
sudo apt-get -y install cusparselt
```
- cudss  
Like cuSPARSELt, NVIDIA provides an installer for us:
```bash
# Download the installer.
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb

# Run the installer and then run the command it spits out.
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb

# Template of what the above command will spit out.
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get -y install cudss
```
2. Install Torch and Torchvision
> Ensure your Python virtual environment is active!
We will be using the wheels made available to us from the [Jetson AI Lab](https://pypi.jetson-ai-lab.io/jp6/cu126) as building from source could take days.
To do this, simply point pip to the Jetson AI Lab package index and install torch and torchvision:
```bash
pip install --force-reinstall --no-cache-dir -U torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```
If for whatever reason that doesn't work, you can try downloading the wheels and then installing from those:
```bash
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226

wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1c0/3de08a69e9554/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=1c03de08a69e95542024477e0cde95fab3436804917133d3f00e67629d3fe902

pip install ./<torch-wheel-file-here>
pip install ./<torchvision-wheel-file-here>
```
> NOTE: You may need to go to [Jetson AI Lab](https://pypi.jetson-ai-lab.io/jp6/cu126) and copy the up-to-date links for torch and torchvision.
#### Test Scripts
Run these test scripts to ensure the GPU can be found and utilized by PyTorch:

Matrix Multiplication (CPU vs GPU timing)
```python
import torch
import time

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Size of test matrices
size = 4000
a = torch.randn(size, size)
b = torch.randn(size, size)

# Run on CPU
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print("CPU matmul time:", time.time() - start, "seconds")

# Run on GPU
if torch.cuda.is_available():
    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")

    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # wait for GPU to finish
    print("GPU matmul time:", time.time() - start, "seconds")
```

Tiny CNN on MNIST
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=512, shuffle=True
)

# Define a simple CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
optimizer = optim.Adam(model.parameters())

# Train 1 epoch (just to test GPU)
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 50 == 0:
        print(f"Train Step: {batch_idx}, Loss: {loss.item()}")
    if batch_idx > 100:  # stop early
        break
```

Ensure that the GPU is found and used.
> You can even see the GPU being utilized via `jtop`!
### TensorFlow
Like we did for the PyTorch installation, we will be using prebuilt wheels as building from source could take days. Unfortunately, the Jetson AI Lab does not (at the time of this instructional write-up) have prebuilt wheels for TensorFlow. However, [NVIDIA does](https://developer.download.nvidia.com/compute/redist/jp/v61/) (and PyTorch wheels as well, but not as updated as the Jetson AI Lab PyTorch wheel).
1. Install needed system packages
```bash
sudo apt-get update

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```
2. Install TensorFlow
> Ensure your Python virtual environment is active!
```bash
# Download the wheel
wget https://developer.download.nvidia.com/compute/redist/jp/v61/tensorflow/tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl

# Install it
pip install ./tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl
```
It's likely that you will see a versioning conflict with numpy; if so, do the following:
```bash
# Uninstall numpy
pip uninstall numpy

# Install compatible numpy version <2.x
# Numpy version 1.26.4 is the latest 1.x version
pip install numpy==1.26.4
```
#### Test Scripts
Run these test scripts to ensure the GPU can be found and utilized by TensorFlow:

Matrix Multiplication (CPU vs GPU timing)
```python
import tensorflow as tf
import time

# Confirm GPU is detected
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Create large random matrices
size = 4000
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# Run on GPU
with tf.device("/GPU:0"):
    start = time.time()
    c = tf.matmul(a, b)
    tf.experimental.numpy.copy(c)  # force compute
    end = time.time()
    print("GPU matmul time:", end - start, "seconds")

# Run on CPU
with tf.device("/CPU:0"):
    start = time.time()
    c = tf.matmul(a, b)
    tf.experimental.numpy.copy(c)  # force compute
    end = time.time()
    print("CPU matmul time:", end - start, "seconds")
```

Tiny CNN on MNIST
```python
import tensorflow as tf

# Load dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Define a simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train briefly (just 1 epoch to check GPU usage)
model.fit(x_train, y_train, epochs=1, batch_size=512)
```

Ensure that the GPU is found and used.
> You can even see the GPU being utilized via `jtop`!
### InsightFace
This requires managing a few dependencies.
1. Install Insightface:
```bash
pip install insightface
```
This installs various dependencies; remove opencv-python-headless:
```bash
pip uninstall -y opencv-python-headless
```
2. Install onnxruntime and onnx:
```bash
# Use the Jetson AI Lab's GPU build
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

pip install onnx
```
3. Upgrade ml-dtypes:
The version of ml-dtypes we are using (0.3.x) for TensorFlow is not compatible with onnxruntime. However, upgrading it doesn't seem to break TensorFlow, so:
```bash
pip install ml-dtypes==0.5.3
```
### OpenCV
We can make use of the Jetson AI Lab's pre-built wheel:
```bash
pip install opencv-contrib-python --no-dep --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```
The `--no-dep` flag avoids changing your numpy version.  
Confirm that GStreamer and v4l/v4l2 are enabled:
```bash
python -c "import cv2; print(cv2.getBuildInformation())"
```
### PyAudio
1. Install needed system packages
```bash
sudo apt-get install libasound-dev portaudio-19-dev libportaudio2 libportaudiocpp0
```
2. Install PyAudio
```bash
pip install PyAudio
```
### Remaining packages
The remaining packages from the [Requirements.txt](./Gemma_demo/Requirements.txt) can simply be pip installed:
```bash
pip install librosa scikit-learn scipy transformers safetensors sounddevice matplotlib
```

⚠️ It's quite possible that numpy was upgraded to numpy>=2.x during the environment setup process.
This can be checked and remedied (ensure your venv is active):
```bash
# Check numpy version
python -c "import numpy; print(numpy.__version__)"
# If output is >=2.x:
pip uninstall numpy
pip install numpy==1.26.4
```
Our venv is intentionally "Frankensteined" because we're on specialized hardware and rely on community-managed wheels rather than professionally maintained ones. Running `pip check` will likely show warnings like:
```bash
albumentations 2.0.8 requires opencv-python-headless, which is not installed.
albucore 0.0.24 requires opencv-python-headless, which is not installed.
tensorflow 2.16.1+nv24.8 has requirement ml-dtypes~=0.3.1, but you have ml-dtypes 0.5.3.
opencv-contrib-python 4.12.0 has requirement numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4.
```
Be aware that when you update or install packages, pip may attempt to resolve these constraints, which can break the environment.
Recommended practices:
- Use `--no-deps` when installing wheels from Jetson AI Lab to avoid dependency churn.
- Re-run the numpy check above after any package change.

It's possible that opencv-python-headless was installed while upgrading or installing packages, if so:
Check and remove opencv-python-headless
To ensure the correct OpenCV build is used (and to avoid headless conflicts), check and remove `opencv-python-headless` if present (ensure your venv is active):
```bash
# Check if opencv-python-headless is installed
pip show opencv-python-headless || echo "opencv-python-headless not installed"

# Remove it if present
pip uninstall -y opencv-python-headless
```
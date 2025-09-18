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
1. To set up the Tapo C120 for use with the Jetson Orin Nano, follow the instructions in the box and install the TP-Link Tapo app.
2. Create a Tapo account and follow the on-screen steps to add the camera.
3. After the camera is added, open it in the app, tap the settings icon (top-right), then go to "Advanced Settings" → "Camera Account" and create a camera account (this is different from your app account).
4. Find your camera's IP address in "Advanced Settings" → "Network Settings". While you're there, turn on "Static IP".
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
We have an Adafruit Feather nRF52840 microcontroller which includes sensors for air temperature, humidity, and light level. To connect to this to the Jetson, simply use a USB to USB-C connector. 

In the ardino environment, save this code:
```bash
/* MODIFIED VERSION
 * Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

 //Use "VOICE START" and "VOICE STOP" to start and stop this program. Includes all sensors including gas.

namespace std {
    void __throw_out_of_range_fmt(const char* fmt, ...) {
        Serial.println("ERROR: Vector out of bounds access detected!");
        Serial.println("Edge Impulse library tried to access invalid vector index");
        Serial.flush(); // Make sure the message gets sent before freezing
        while(1) {
            delay(1000); // Stop execution
        }
    }
}

static const int PDM_DATA_PIN  = 34;
static const int PDM_CLOCK_PIN = 35;
static const int PDM_POWER_PIN = -1;    // no power-enable pin

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 2

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <speech-recognition-correct-words_inferencing.h>
#include <Adafruit_APDS9960.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_LSM6DS33.h>
#include <Adafruit_LSM6DS3TRC.h>
#include <Adafruit_SHT31.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include "ScioSense_ENS160.h"  // ENS160 library

// Define variables
Adafruit_APDS9960 apds9960; // proximity, light, color, gesture
Adafruit_BMP280 bmp280;     // temperautre, barometric pressure
Adafruit_LIS3MDL lis3mdl;   // magnetometer
Adafruit_LSM6DS3TRC lsm6ds3trc; // accelerometer, gyroscope
Adafruit_LSM6DS33 lsm6ds33;
Adafruit_SHT31 sht30;       // humidity
ScioSense_ENS160 ens160(ENS160_I2CADDR_1); // air quality sensor
bool was_recording = false;

uint8_t proximity;
uint16_t r, g, b, c;
float temperature, pressure, altitude;
float magnetic_x, magnetic_y, magnetic_z;
float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;
float humidity;
int32_t mic;
uint16_t aqi, tvoc, eco2;  // ENS160 readings
long int accel_array[6];
long int check_array[6]={0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
bool is_recording = false;
int sensor_counter;
bool ens160_available = false;  // Track ENS160 status

bool new_rev = true;

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

static const int led_pin = LED_BUILTIN;

/**
 * @brief      Arduino setup function
 */
void setup()
{
    pinMode(led_pin, OUTPUT);
   
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse + ENS160 Inferencing Demo");
   
    // initialize the sensors
    apds9960.begin();
    apds9960.enableProximity(true);
    apds9960.enableColor(true);
    bmp280.begin();
    lis3mdl.begin_I2C();
    lsm6ds33.begin_I2C();

    // check for readings from LSM6DS33
    sensors_event_t accel;
    sensors_event_t gyro;
    sensors_event_t temp;
    lsm6ds33.getEvent(&accel, &gyro, &temp);
    accel_array[0] = accel.acceleration.x;
    accel_array[1] = accel.acceleration.y;
    accel_array[2] = accel.acceleration.z;
    accel_array[3] = gyro.gyro.x;
    accel_array[4] = gyro.gyro.y;
    accel_array[5] = gyro.gyro.z;
    // if all readings are empty, then new rev
    for (int i =0; i < 5; i++) {
      if (accel_array[i] != check_array[i]) {
        new_rev = false;
        break;
      }
    }
   
    // and we need to instantiate the LSM6DS3TRC
    if (new_rev) {
      lsm6ds3trc.begin_I2C();
    }
    sht30.begin();

    // Initialize ENS160
    Serial.print("ENS160...");
    bool ens160_ok = ens160.begin();
    ens160_available = ens160.available();
    Serial.println(ens160_available ? "done." : "failed!");
   
    if (ens160_available) {
        // Print ENS160 versions
        Serial.print("\tRev: ");
        Serial.print(ens160.getMajorRev());
        Serial.print(".");
        Serial.print(ens160.getMinorRev());
        Serial.print(".");
        Serial.println(ens160.getBuild());
       
        // Set to standard operating mode
        Serial.print("\tStandard mode ");
        Serial.println(ens160.setMode(ENS160_OPMODE_STD) ? "done." : "failed!");
       
        // Set initial environmental data
        ens160.set_envdata(25.0, 50.0);
       
        Serial.println("ENS160 warming up (30 seconds)...");
        delay(30000);  // 30 second warm-up for ENS160
        Serial.println("ENS160 ready!");
    } else {
        Serial.println("ENS160 initialization failed - continuing without air quality data");
    }

    // tell the PDM driver which pins to use
    PDM.setPins(PDM_DATA_PIN, PDM_CLOCK_PIN, PDM_POWER_PIN);
   
    PDM.setBufferSize(512);  // Smaller buffer
    PDM.setGain(20);         // Lower gain
    if (!PDM.begin(1, 16000)) {  // Lower sample rate
      Serial.println("Failed to start PDM!");
      while(1);
    }

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }
   
    Serial.println("System ready! Say your wake words to start/stop data transmission.");
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // Start or stop sending data if value of word is above a threshold
    bool should_record = is_recording;
    if (result.classification[2].value > 0.7){
      should_record = true;
      //Serial.println("*** START command detected - Beginning sensor data transmission ***");
    }
    if (result.classification[3].value > 0.7){
      should_record = false;
      //Serial.println("*** STOP command detected - Ending sensor data transmission ***");
    }

    // Edge-detect: only print when the state changes
    if (should_record && !was_recording){
      is_recording = true;
      was_recording = true;
      Serial.println("START");
      // CSV header, once per session:
      Serial.println("HEADER,time_ms,red,green,blue,clear,temperatureC,humidityPct,accelZ_mps2,aqi,tvoc_ppb,eco2_ppm");
    }
    else if (!should_record && was_recording){
      is_recording = false;
      was_recording = false;
      Serial.println("STOP");
    }
   
   
    if (is_recording) {
        sensor_counter++;
        if (sensor_counter >= 3) {  // Only send every 3rd cycle
            send_sensor_data();
            sensor_counter = 0;
        }
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        // Optionally show classification confidence for debugging
        /*
        ei_printf("Predictions: ");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("%s: %.2f ", result.classification[ix].label,
                      result.classification[ix].value);
        }
        ei_printf("\n");
        */
       
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

        print_results = 0;
    }
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>>1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    record_ready = true;

    return true;
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable) {
    int bytesRead = PDM.read(sampleBuffer, bytesAvailable);
    if (bytesRead != bytesAvailable) {
      Serial.println("PDM read mismatch!");
    }
  }
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

static void send_sensor_data(){
  // Read color sensor data
  while (!apds9960.colorDataReady()) {
    delay(5);
  }
  apds9960.getColorData(&r, &g, &b, &c);

  // Read temperature and pressure
  temperature = bmp280.readTemperature();

  // Read magnetometer
  lis3mdl.read();

  // Read accelerometer/gyroscope
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  if (new_rev) {
    lsm6ds3trc.getEvent(&accel, &gyro, &temp);
  }
  else {
    lsm6ds33.getEvent(&accel, &gyro, &temp);
  }
  accel_z = accel.acceleration.z;

  // Read humidity
  humidity = sht30.readHumidity();

  // Read ENS160 air quality data
  if (ens160_available) {
    // Update environmental data with real sensor readings
    ens160.set_envdata(temperature, humidity);
   
    // Perform measurement
    ens160.measure();
   
    // Get readings
    aqi = ens160.getAQI();
    tvoc = ens160.getTVOC();
    eco2 = ens160.geteCO2();
  } else {
    // Set error values if sensor not available
    aqi = 999;
    tvoc = 65535;
    eco2 = 65535;
  }

  // Print all sensor data
  unsigned long t_ms = millis();
  Serial.print("DATA,");
  Serial.print(t_ms);             Serial.print(',');
  Serial.print(r);                Serial.print(',');
  Serial.print(g);                Serial.print(',');
  Serial.print(b);                Serial.print(',');
  Serial.print(c);                Serial.print(',');
  Serial.print(temperature, 2);   Serial.print(',');
  Serial.print(humidity, 2);      Serial.print(',');
  Serial.print(accel_z, 3);       Serial.print(',');
  Serial.print(aqi);              Serial.print(',');
  Serial.print(tvoc);             Serial.print(',');
  Serial.println(eco2);
  /*
  Serial.println("\nFeather Sense + ENS160 Sensor Information");
  Serial.println("---------------------------------------------");
  Serial.print("Red: ");
  Serial.print(r);
  Serial.print(" Green: ");
  Serial.print(g);
  Serial.print(" Blue: ");
  Serial.print(b);
  Serial.print(" Clear: ");
  Serial.println(c);
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" C");
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");
  Serial.print("Acceleration Z: ");
  Serial.print(accel_z);
  Serial.println(" m/s^2");
 
  // Print ENS160 air quality data
  Serial.print("Air Quality Index (AQI): ");
  if (aqi == 255 || aqi == 999) {
    Serial.println("ERROR");
  } else {
    Serial.println(aqi);
  }
  Serial.print("TVOC: ");
  if (tvoc == 65535) {
    Serial.println("ERROR ppb");
  } else {
    Serial.print(tvoc);
    Serial.println(" ppb");
  }
  Serial.print("eCO2: ");
  if (eco2 == 65535) {
    Serial.println("ERROR ppm");
  } else {
    Serial.print(eco2);
    Serial.println(" ppm");
  }*/
 
  delay(50);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
```

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

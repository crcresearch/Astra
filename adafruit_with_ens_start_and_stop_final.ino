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

  delay(50);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif

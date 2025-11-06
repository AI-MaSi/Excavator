#include "sh2_paketti.h"
#include <time.h>
#include "sh2_SensorValue.h"
#include <string.h>
// I2C configuration
#define I2C_PORT i2c0
#define I2C_SDA 0 // GPIO4 (Pico pin 6)
#define I2C_SCL 1  // GPIO5 (Pico pin 7) 
#define I2C_BAUD 400000  // 400 kHz
#define BNO08X_ADDR 0x4A  // BNO08x I2C address

sh2_vector_list_t sh2_vector_list = {
    .cursor = 0,
    .data_ready = false
};

static sh2_Hal_t hal;
static bool reset_received = false;
static int rc;

// This is needed for some reason to avoid some peculiar bugs
static void clear_i2c_flags() {
    i2c_hw_t *i2c = i2c_get_hw(I2C_PORT);
    i2c->enable = 0;  // Disable first for safety
    (void)i2c->clr_tx_abrt;
    (void)i2c->clr_rd_req;
    (void)i2c->clr_tx_over;
    (void)i2c->clr_rx_done;
    (void)i2c->clr_rx_over;
    (void)i2c->clr_rx_under;
    (void)i2c->clr_activity;  // If ACTIVITY_BITS include master/slave activity, clear them
    i2c->enable = 1;  // Re-enable
}


// HAL: Initialize I2C0
static int i2c_open(sh2_Hal_t* pInstance) {
    i2c_init(I2C_PORT, I2C_BAUD);
    gpio_set_function(I2C_SDA, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA);
    gpio_pull_up(I2C_SCL);
    return SH2_OK;
}

// HAL: Close I2C
static void i2c_close(sh2_Hal_t* pInstance) {
    i2c_deinit(I2C_PORT);
}

// HAL: Read I2C data
static int i2c_read(sh2_Hal_t* pInstance, uint8_t *pData, unsigned len, uint32_t *pTimestamp_us) {
    clock_t start_time = clock();

    // Wait for bus to be completely idle
    // while (i2c_get_hw(I2C_PORT)->status & I2C_IC_STATUS_ACTIVITY_BITS) {
    //     tight_loop_contents();
    // }

    clear_i2c_flags();

    int rc = i2c_read_blocking(I2C_PORT, BNO08X_ADDR, pData, len, false);
    float elapsed_time = clock() - start_time;
    printf("Lukemisen aika: %.2f", elapsed_time);
    if (rc != len) return SH2_ERR;
    *pTimestamp_us = to_us_since_boot(get_absolute_time());
    return rc;
}

// HAL: Write I2C data
static int i2c_write(sh2_Hal_t* pInstance, uint8_t *pData, unsigned len) {
    clock_t start_time = clock();
    // Wait for bus to be completely idle
    // while (i2c_get_hw(I2C_PORT)->status & I2C_IC_STATUS_ACTIVITY_BITS) {
    //     tight_loop_contents();
    // }

    clear_i2c_flags();

    int result = i2c_write_blocking(I2C_PORT, BNO08X_ADDR, pData, len, false);
    float elapsed_time = clock() - start_time;
    printf("Kirjoituksen aika: %.2f", elapsed_time);
    return (result == len) ? result : SH2_ERR;
}

// HAL: Get microsecond timestamp
static uint32_t get_time_us(sh2_Hal_t* pInstance) {
    return to_us_since_boot(get_absolute_time());
}

// Sensor event callback
static void sensor_handler(void *cookie, sh2_SensorEvent_t *event) {
    uint16_t local_cursor = sh2_vector_list.cursor;
    #define l sh2_vector_list.rolling_list
    sh2_SensorValue_t value;
    if (sh2_decodeSensorEvent(&value, event) == SH2_OK) {
        if (value.sensorId == SH2_ROTATION_VECTOR) {
            if (sh2_vector_list.cursor >= SH_2_VECOTR_LIST_ROW_MAX || !sh2_vector_list.data_ready) {
                local_cursor = 0;
            }
            else {
                local_cursor++;
            }
            l[local_cursor][0] = value.un.rotationVector.real;
            l[local_cursor][1] = value.un.rotationVector.i;
            l[local_cursor][2] = value.un.rotationVector.j;
            l[local_cursor][3] = value.un.rotationVector.k;
            sh2_vector_list.cursor = local_cursor;
            if(sh2_vector_list.data_ready == false) 
            {
               sh2_vector_list.data_ready = true; 
            }
        }
    } else {
        printf("wrong event?\n");
    }
    #undef l
}

// Async event callback
static void async_handler(void *cookie, sh2_AsyncEvent_t *event) {
    if (event->eventId == SH2_RESET) {
        printf("BNO08x reset\n");
        reset_received = true;
    }
}

static void sh2_open_or_halt() {
    // Open SH-2 interface
    rc = sh2_open(&hal, async_handler, NULL);
    if (rc != SH2_OK) {
        printf("SH-2 open failed: %d\n", rc);
        while (1);
    }
}

void printEventWrapper(void * cookie, sh2_SensorEvent_t* event) {
    printEvent(event);
}

static void sh2_setSensorCallback_or_halt() {
    rc = sh2_setSensorCallback(printEventWrapper, NULL);
    if (rc != SH2_OK) {
        printf("Callback setup failed: %d\n", rc);
        sh2_close();
        while (1);
    }
}

static void sh2_devReset_or_halt() {
    rc = sh2_devReset();
    if (rc != SH2_OK) {
        printf("Reset failed: %d\n", rc);
        sh2_close();
        while (1);
    }
    reset_received = false;
    sleep_ms(300);// Wait for stabilization
}

// Print a sensor event to the console
void printEvent(const sh2_SensorEvent_t * event)
{
    int rc;
    sh2_SensorValue_t value;
    float scaleRadToDeg = 180.0 / 3.14159265358;
    float r, i, j, k, acc_deg, x, y, z;
    float t;
    static int skip = 0;

    if(sh2_vector_list.data_ready == false) 
    {
        sh2_vector_list.data_ready = true; 
    }

    rc = sh2_decodeSensorEvent(&value, event);
    if (rc != SH2_OK) {
        printf("Error decoding sensor event: %d\n", rc);
        return;
    }

    t = value.timestamp / 1000000.0;  // time in seconds.
    switch (value.sensorId) {
        case SH2_RAW_ACCELEROMETER:
            printf("%8.4f Raw acc: %d %d %d time_us:%d\n",
                   (double)t,
                   value.un.rawAccelerometer.x,
                   value.un.rawAccelerometer.y,
                   value.un.rawAccelerometer.z,
                   value.un.rawAccelerometer.timestamp);
            break;

        case SH2_ACCELEROMETER:
            printf("%8.4f Acc: %f %f %f\n",
                   (double)t,
                   (double)value.un.accelerometer.x,
                   (double)value.un.accelerometer.y,
                   (double)value.un.accelerometer.z);
            break;
            
        case SH2_RAW_GYROSCOPE:
            printf("%8.4f Raw gyro: x:%d y:%d z:%d temp:%d time_us:%d\n",
                   (double)t,
                   value.un.rawGyroscope.x,
                   value.un.rawGyroscope.y,
                   value.un.rawGyroscope.z,
                   value.un.rawGyroscope.temperature,
                   value.un.rawGyroscope.timestamp);
            break;
            
        case SH2_ROTATION_VECTOR:
            r = value.un.rotationVector.real;
            i = value.un.rotationVector.i;
            j = value.un.rotationVector.j;
            k = value.un.rotationVector.k;
            acc_deg = scaleRadToDeg * 
                value.un.rotationVector.accuracy;
            printf("%8.4f Rotation Vector: "
                   "r:%0.6f i:%0.6f j:%0.6f k:%0.6f (acc: %0.6f deg)\n",
                   (double)t,
                   (double)r, (double)i, (double)j, (double)k, (double)acc_deg);
            break;
        case SH2_GAME_ROTATION_VECTOR:
            r = value.un.gameRotationVector.real;
            i = value.un.gameRotationVector.i;
            j = value.un.gameRotationVector.j;
            k = value.un.gameRotationVector.k;
            printf("%8.4f GRV: "
                   "r:%0.6f i:%0.6f j:%0.6f k:%0.6f\n",
                   (double)t,
                   (double)r, (double)i, (double)j, (double)k);
            break;
        case SH2_GYROSCOPE_CALIBRATED:
            x = value.un.gyroscope.x;
            y = value.un.gyroscope.y;
            z = value.un.gyroscope.z;
            printf("%8.4f GYRO: "
                   "x:%0.6f y:%0.6f z:%0.6f\n",
                   (double)t,
                   (double)x, (double)y, (double)z);
            break;
        case SH2_GYROSCOPE_UNCALIBRATED:
            x = value.un.gyroscopeUncal.x;
            y = value.un.gyroscopeUncal.y;
            z = value.un.gyroscopeUncal.z;
            printf("%8.4f GYRO_UNCAL: "
                   "x:%0.6f y:%0.6f z:%0.6f\n",
                   (double)t,
                   (double)x, (double)y, (double)z);
            break;
        case SH2_GYRO_INTEGRATED_RV:
            // These come at 1kHz, too fast to print all of them.
            // So only print every 10th one
            skip++;
            if (skip == 10) {
                skip = 0;
                r = value.un.gyroIntegratedRV.real;
                i = value.un.gyroIntegratedRV.i;
                j = value.un.gyroIntegratedRV.j;
                k = value.un.gyroIntegratedRV.k;
                x = value.un.gyroIntegratedRV.angVelX;
                y = value.un.gyroIntegratedRV.angVelY;
                z = value.un.gyroIntegratedRV.angVelZ;
                printf("%8.4f Gyro Integrated RV: "
                       "r:%0.6f i:%0.6f j:%0.6f k:%0.6f x:%0.6f y:%0.6f z:%0.6f\n",
                       (double)t,
                       (double)r, (double)i, (double)j, (double)k,
                       (double)x, (double)y, (double)z);
            }
            break;
        case SH2_IZRO_MOTION_REQUEST:
            printf("IZRO Request: intent:%d, request:%d\n",
                   value.un.izroRequest.intent,
                   value.un.izroRequest.request);
            break;
        case SH2_SHAKE_DETECTOR:
            printf("Shake Axis: %c%c%c\n", 
                   (value.un.shakeDetector.shake & SHAKE_X) ? 'X' : '.',
                   (value.un.shakeDetector.shake & SHAKE_Y) ? 'Y' : '.',
                   (value.un.shakeDetector.shake & SHAKE_Z) ? 'Z' : '.');

            break;
        case SH2_STABILITY_CLASSIFIER:
            printf("Stability Classification: %d\n",
                   value.un.stabilityClassifier.classification);
            break;
        case SH2_STABILITY_DETECTOR:
            printf("Stability Detector: %d\n",
                   value.un.stabilityDetector.stability);
            break;
        default:
            printf("Unknown sensor: %d\n", value.sensorId);
            break;
    }
}

static void sh2_setSensorConfig_or_halt() {
    // Enable rotation vector (100 Hz)
    sh2_SensorConfig_t config;
    memset(&config, 0, sizeof(config));
    float test = (1.0f/(float)SAMPLE_RATE) * 1000000.0f; // DANGER
    config.reportInterval_us = test; // 10 ms = 100 Hz
    // TODO - add a global array rolling array and only take the latest value

    rc = sh2_setSensorConfig(SH2_RAW_GYROSCOPE, &config);
    if (rc != SH2_OK) {
        printf("Config failed: %d\n", rc);
        sh2_close();
        while (1);
    }
}

static void initialize_HALL() {
    hal.open = i2c_open;
    hal.close = i2c_close;
    hal.read = i2c_read;
    hal.write = i2c_write;
    hal.getTimeUs = get_time_us;
}

static void wait_for_reset_or_halt() {
    absolute_time_t timeout = make_timeout_time_ms(2000);
    while (!reset_received) {
        sh2_service();
        if (time_reached(timeout)) {
            printf("Timeout waiting for reset event!\n");
            sh2_close();
            while(1);
        }
        sleep_ms(10);
    }
}

void setup_sh2_service() {
    initialize_HALL();
    sh2_open_or_halt();
    sh2_setSensorCallback_or_halt();
    // most likely unneeded because sh2 open already does software reset
    sh2_devReset_or_halt(); 
    wait_for_reset_or_halt();
    sh2_setSensorConfig_or_halt();
}

void read_super_sensor() {
    sh2_service();
}

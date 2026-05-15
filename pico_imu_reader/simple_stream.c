#include "Fusion.h"
#include "InitFusion.h"
#include "i2c_helpers.h"
#include "ism330dlc.h"
#include "output.h"
#include "settings.h"
#include "status_led.h"
#include "tusb.h"
#include <math.h>
#include <pico/time.h>
#include <pico/stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#define SIMPLE_STREAM_SENSOR_COUNT 4u
#define SIMPLE_STREAM_RATE_HZ 416
#define SIMPLE_STREAM_GYRO_DPS 250.0f
#define SIMPLE_STREAM_AHRS_GAIN 5.0f
#define SIMPLE_STREAM_ACCEL_REJECTION 20.0f
#define SIMPLE_STREAM_RECOVERY_S 0.5f

static inline FusionQuaternion enforce_quaternion_continuity(FusionQuaternion q, FusionQuaternion q_ref) {
    const float dot = q.element.w * q_ref.element.w +
                      q.element.x * q_ref.element.x +
                      q.element.y * q_ref.element.y +
                      q.element.z * q_ref.element.z;

    if (dot < 0.0f) {
        q.element.w = -q.element.w;
        q.element.x = -q.element.x;
        q.element.y = -q.element.y;
        q.element.z = -q.element.z;
    }
    return q;
}

static inline float clamp_delta_time(float delta_time_s) {
    const float nominal = 1.0f / (float)SIMPLE_STREAM_RATE_HZ;
    if (!isfinite(delta_time_s) || (delta_time_s <= 0.0f)) {
        return nominal;
    }
    if (delta_time_s < nominal * 0.5f) {
        return nominal * 0.5f;
    }
    if (delta_time_s > nominal * 1.5f) {
        return nominal * 1.5f;
    }
    return delta_time_s;
}

static void service_usb_until(uint64_t deadline_us) {
    while (time_us_64() < deadline_us) {
        tud_task();

        const uint64_t now_us = time_us_64();
        if (now_us >= deadline_us) {
            return;
        }

        const uint64_t remaining_us = deadline_us - now_us;
        if (remaining_us > 1000u) {
            sleep_us(500);
        } else if (remaining_us > 100u) {
            sleep_us((uint32_t)(remaining_us - 50u));
        } else {
            tight_loop_contents();
        }
    }
}

static void apply_simple_settings(void) {
    imu_reader_settings.sampleRate = SIMPLE_STREAM_RATE_HZ;
    imu_reader_settings.gyroRangeDps = SIMPLE_STREAM_GYRO_DPS;
    imu_reader_settings.ahrsGain = SIMPLE_STREAM_AHRS_GAIN;
    imu_reader_settings.ahrsAccelRejection = SIMPLE_STREAM_ACCEL_REJECTION;
    imu_reader_settings.ahrsRecoveryPeriodS = SIMPLE_STREAM_RECOVERY_S;
}

static void stream_forever(Sensor *sensors, uint8_t sensor_count) {
    float sensors_data[MAX_SENSORS][FLOATS_PER_SENSOR] = {{0.0f}};
    bool sensor_valid[MAX_SENSORS] = {false};
    const uint64_t period_us = 1000000u / SIMPLE_STREAM_RATE_HZ;
    uint64_t next_loop_us = time_us_64();
    uint64_t previous_loop_start_us = next_loop_us;

    while (true) {
        const uint64_t loop_start_us = time_us_64();
        const float delta_time_s = clamp_delta_time((float)(loop_start_us - previous_loop_start_us) / 1000000.0f);
        previous_loop_start_us = loop_start_us;

        tud_task();
        read_all_sensors(sensors, sensor_valid);

        for (uint8_t i = 0; i < sensor_count; i++) {
            if (!sensor_valid[i]) {
                sensors_data[i][0] = sensors[i].previousQuaternion.element.w;
                sensors_data[i][1] = sensors[i].previousQuaternion.element.x;
                sensors_data[i][2] = sensors[i].previousQuaternion.element.y;
                sensors_data[i][3] = sensors[i].previousQuaternion.element.z;
                sensors_data[i][4] = 0.0f;
                sensors_data[i][5] = 0.0f;
                sensors_data[i][6] = 0.0f;
                continue;
            }

            FusionVector gyroscope = FusionCalibrationInertial(
                sensors[i].gyroscope,
                sensors[i].calibration.gyroscopeMisalignment,
                sensors[i].calibration.gyroscopeSensitivity,
                sensors[i].calibration.gyroscopeOffset);

            FusionVector accelerometer = FusionCalibrationInertial(
                sensors[i].accelerometer,
                sensors[i].calibration.accelerometerMisalignment,
                sensors[i].calibration.accelerometerSensitivity,
                sensors[i].calibration.accelerometerOffset);

            gyroscope = FusionOffsetUpdate(&sensors[i].offset, gyroscope);
            FusionAhrsUpdate(&sensors[i].ahrs, gyroscope, accelerometer, FUSION_VECTOR_ZERO, delta_time_s);

            FusionQuaternion quaternion = FusionAhrsGetQuaternion(&sensors[i].ahrs);
            quaternion = enforce_quaternion_continuity(quaternion, sensors[i].previousQuaternion);
            sensors[i].previousQuaternion = quaternion;

            sensors_data[i][0] = quaternion.element.w;
            sensors_data[i][1] = quaternion.element.x;
            sensors_data[i][2] = quaternion.element.y;
            sensors_data[i][3] = quaternion.element.z;
            sensors_data[i][4] = gyroscope.axis.x;
            sensors_data[i][5] = gyroscope.axis.y;
            sensors_data[i][6] = gyroscope.axis.z;
        }

        print_output_data(loop_start_us, sensors_data, sensor_count);

        next_loop_us += period_us;
        const uint64_t now_us = time_us_64();
        if (now_us < next_loop_us) {
            service_usb_until(next_loop_us);
        } else if ((now_us - next_loop_us) > period_us) {
            next_loop_us = now_us;
        }
    }
}

static void error_forever(uint8_t msg_type) {
    status_led_set(STATUS_ERROR);
    send_control_msg(msg_type);
    while (true) {
        tud_task();
        sleep_ms(50);
    }
}

int main(void) {
    tusb_init();
    status_led_init();
    status_led_set(STATUS_INIT);

    apply_simple_settings();

    if (!setup_I2C_pins(400*1000)) {
        error_forever(MSG_TYPE_ERR_I2C);
    }

    if (!initialize_sensors(416)) {
        error_forever(MSG_TYPE_ERR_IMU);
    }

    const uint8_t sensor_count = get_active_sensor_count();
    if (sensor_count != SIMPLE_STREAM_SENSOR_COUNT) {
        error_forever(MSG_TYPE_ERR_IMU);
    }

    Sensor sensors[MAX_SENSORS];
    initialize_sensors_values(sensors, MAX_SENSORS);
    initialize_calibrations(sensors, MAX_SENSORS);
    initialize_algos(sensors, MAX_SENSORS);

    status_led_set(STATUS_STREAM);
    stream_forever(sensors, sensor_count);
    return 0;
}

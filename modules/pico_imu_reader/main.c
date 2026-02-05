#include "i2c_helpers.h"
#include "InitFusion.h"
#include "Fusion.h"
#include "ism330dlc.h"
#include "tusb.h"
#include "cdc_console.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "settings.h"
#include "output.h"
#include <math.h>

// NUM_SENSORS and FLOATS_PER_SENSOR defined in output.h

static inline FusionQuaternion enforce_quaternion_continuity(FusionQuaternion q, FusionQuaternion q_ref) {
    // Choose hemisphere (q or -q) closest to reference quaternion
    float dot = q.element.w * q_ref.element.w +
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

static inline float unwrap_angle_deg(float current, float last, float *offset) {
    float delta = current - last;
    if (delta > 180.0f) {
        *offset -= 360.0f;
    } else if (delta < -180.0f) {
        *offset += 360.0f;
    }
    return current + *offset;
}

static inline float quaternion_y_twist_deg(const FusionQuaternion q_in) {
    // Extract Y-axis twist: angle = 2*atan2(y, w) mapped to (-180, 180]
    float w = q_in.element.w;
    float y = q_in.element.y;
    float angle_deg = 2.0f * (180.0f / (float)M_PI) * atan2f(y, w);
    if (angle_deg <= -180.0f) angle_deg += 360.0f;
    else if (angle_deg > 180.0f) angle_deg -= 360.0f;
    return angle_deg;
}

// Main sensor loop - clean AHRS, no software filtering
void update_loop(float period_ms, float sensors_data[][FLOATS_PER_SENSOR], Sensor* sensors) {
    while (true) {
        tud_task();
        uint64_t loop_start = time_us_64();

        read_all_sensors(sensors);

        for (int i = 0; i < NUM_SENSORS; i++) {
            // Apply calibration (identity by default)
            sensors[i].gyroscope = FusionCalibrationInertial(
                sensors[i].gyroscope,
                sensors[i].calibration.gyroscopeMisalignment,
                sensors[i].calibration.gyroscopeSensitivity,
                sensors[i].calibration.gyroscopeOffset);

            sensors[i].accelerometer = FusionCalibrationInertial(
                sensors[i].accelerometer,
                sensors[i].calibration.accelerometerMisalignment,
                sensors[i].calibration.accelerometerSensitivity,
                sensors[i].calibration.accelerometerOffset);

            // Gyro offset tracking (stationary bias removal)
            sensors[i].gyroscope = FusionOffsetUpdate(&sensors[i].offset, sensors[i].gyroscope);

            // Compute delta time
            const float deltaTime = (float)(sensors[i].timestamp - sensors[i].previousTimestamp) / 1e6f;
            sensors[i].previousTimestamp = sensors[i].timestamp;

            // AHRS update (no magnetometer, heading fixed at 0)
            FusionAhrsUpdateExternalHeading(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, 0.0f, deltaTime);

            // Get quaternion and enforce continuity
            FusionQuaternion quat = FusionAhrsGetQuaternion(&sensors[i].ahrs);
            quat = enforce_quaternion_continuity(quat, sensors[i].previousQuaternion);
            sensors[i].previousQuaternion = quat;

            // Track pitch (for re-zero feature)
            float pitchDeg = quaternion_y_twist_deg(quat);
            float unwrapped = unwrap_angle_deg(pitchDeg, sensors[i].lastPitchDeg, &sensors[i].unwrapOffsetDeg);
            sensors[i].lastPitchDeg = pitchDeg;
            sensors[i].pitchDeg = unwrapped;

            // Output: quaternion + gyro
            sensors_data[i][0] = quat.element.w;
            sensors_data[i][1] = quat.element.x;
            sensors_data[i][2] = quat.element.y;
            sensors_data[i][3] = quat.element.z;
            sensors_data[i][4] = sensors[i].gyroscope.axis.x;
            sensors_data[i][5] = sensors[i].gyroscope.axis.y;
            sensors_data[i][6] = sensors[i].gyroscope.axis.z;
        }

        // Send binary frame
        uint64_t out_ts = time_us_64();
        print_output_data(out_ts, sensors_data);

        // Check for runtime ZERO command
        check_for_zero_command();
        if (zero_requested) {
            for (int i = 0; i < NUM_SENSORS; i++) {
                sensors[i].zeroOffsetDeg = sensors[i].pitchDeg;
            }
            zero_requested = 0;
            cdc_write_line("Re-zero applied.");
        }

        // Maintain target loop rate
        uint64_t elapsed_us = time_us_64() - loop_start;
        uint64_t target_us = (uint64_t)(period_ms * 1000.0f);
        if (elapsed_us < target_us) {
            sleep_us((uint32_t)(target_us - elapsed_us));
        }
    }
}

int main() {
    stdio_init_all();

    // Wait for USB CDC connection (poll TinyUSB task)
    while (!tud_cdc_connected()) {
        tud_task();
        sleep_ms(10);
    }

    // Get configuration from host
    wait_for_settings();

    // Setup I2C buses
    if (!setup_I2C_pins()) {
        cdc_write_line("I2C setup failed!");
        return 1;
    }

    // Initialize IMU hardware
    if (!initialize_sensors()) {
        cdc_write_line("IMU init failed!");
        return 1;
    }

    // Setup sensor state and AHRS
    Sensor sensors[NUM_SENSORS];
    initialize_sensors_values(sensors, NUM_SENSORS);
    initialize_calibrations(sensors, NUM_SENSORS);
    initialize_algos(sensors, NUM_SENSORS);

    // Compute loop period from sample rate
    float period_ms = 1000.0f / (float)imu_reader_settings.sampleRate;

    // Output buffer
    float sensors_data[NUM_SENSORS][FLOATS_PER_SENSOR];

    cdc_writef("Starting AHRS loop @ %d Hz\n", imu_reader_settings.sampleRate);
    // Disable text output during binary streaming to avoid interleaving.
    cdc_console_enable(false);
    update_loop(period_ms, sensors_data, sensors);

    return 0;
}

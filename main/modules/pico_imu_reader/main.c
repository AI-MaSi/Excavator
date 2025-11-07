#include "i2c_helpers.h"
#include "InitFusion.h"
#include "Fusion.h"
#include "ism330dlc.h"
#include "tusb.h"
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "settings.h"
#include "output.h"
#include <math.h>

// extern sh2_vector_list_t sh2_vector_list;

#define LOOP_DURATION_MS 6.252f

static inline FusionQuaternion enforce_quaternion_continuity(FusionQuaternion q, FusionQuaternion q_ref) {
    // Choose hemisphere (q or -q) closest to reference quaternion
    // Compute dot product to measure distance between quaternions
    float dot = q.element.w * q_ref.element.w +
                q.element.x * q_ref.element.x +
                q.element.y * q_ref.element.y +
                q.element.z * q_ref.element.z;

    // If dot product is negative, quaternions are in opposite hemispheres
    // Flip to the closer hemisphere
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
    // Map to (-180, 180]
    if (angle_deg <= -180.0f) angle_deg += 360.0f;
    else if (angle_deg > 180.0f) angle_deg -= 360.0f;
    return angle_deg;
}

static inline FusionQuaternion apply_mounting(const FusionQuaternion mount, const FusionQuaternion sensor) {
    // mounted = mount * sensor
    return FusionQuaternionMultiply(mount, sensor);
}

float apply_lpf(float new_val, float old_value) {
    return old_value*(1.0f-imu_reader_settings.lpf_alpha) + imu_reader_settings.lpf_alpha*new_val;
}

void update_loop_with_lpf(float sleep_time, float sensors_data[][11], Sensor* sensors) {
    while (true) {
    uint64_t loop_start = time_us_64();
    read_all_sensors(sensors);
    for (int i=0; i<imu_reader_settings.sensorCount;i++) {
        sensors[i].gyroscope_old.axis.x = apply_lpf(sensors[i].gyroscope.axis.x, sensors[i].gyroscope_old.axis.x);
        sensors[i].gyroscope_old.axis.y = apply_lpf(sensors[i].gyroscope.axis.y, sensors[i].gyroscope_old.axis.y);
        sensors[i].gyroscope_old.axis.z = apply_lpf(sensors[i].gyroscope.axis.z, sensors[i].gyroscope_old.axis.z);

        sensors[i].accelerometer_old.axis.x = apply_lpf(sensors[i].accelerometer.axis.x, sensors[i].accelerometer_old.axis.x);
        sensors[i].accelerometer_old.axis.y = apply_lpf(sensors[i].accelerometer.axis.y, sensors[i].accelerometer_old.axis.y);
        sensors[i].accelerometer_old.axis.z = apply_lpf(sensors[i].accelerometer.axis.z, sensors[i].accelerometer_old.axis.z);

        sensors[i].gyroscope = FusionCalibrationInertial(sensors[i].gyroscope_old, sensors[i].calibration.gyroscopeMisalignment, sensors[i].calibration.gyroscopeSensitivity, sensors[i].calibration.gyroscopeOffset);
        sensors[i].accelerometer = FusionCalibrationInertial(sensors[i].accelerometer_old, sensors[i].calibration.accelerometerMisalignment, sensors[i].calibration.accelerometerSensitivity, sensors[i].calibration.accelerometerOffset);
        sensors[i].gyroscope = FusionOffsetUpdate(&sensors[i].offset, sensors[i].gyroscope);

        const float deltaTime = (float) (sensors[i].timestamp - sensors[i].previousTimestamp) / 1e6f;
        sensors[i].previousTimestamp = sensors[i].timestamp;
        FusionAhrsUpdateExternalHeading(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, 0.0f, deltaTime);
        FusionQuaternion quat = FusionAhrsGetQuaternion(&sensors[i].ahrs);
        quat = enforce_quaternion_continuity(quat, sensors[i].previousQuaternion);
        sensors[i].previousQuaternion = quat;  // Store for next iteration
        quat = apply_mounting(sensors[i].mountingQuaternion, quat);
        // Track pitch and optionally enforce pitch-only quaternion when requested
        float pitchDeg = quaternion_y_twist_deg(quat);
        float unwrapped = unwrap_angle_deg(pitchDeg, sensors[i].lastPitchDeg, &sensors[i].unwrapOffsetDeg);
        sensors[i].lastPitchDeg = pitchDeg;
        sensors[i].pitchDeg = unwrapped - sensors[i].zeroOffsetDeg;
        if (imu_reader_settings.pitchOnly) {
            const float rad = sensors[i].pitchDeg * ((float)M_PI / 180.0f);
            const float half = 0.5f * rad;
            const float c = cosf(half);
            const float s = sinf(half);
            quat.element.w = c;
            quat.element.x = 0.0f;
            quat.element.y = s;
            quat.element.z = 0.0f;
        } else if (sensors[i].zeroOffsetDeg != 0.0f) {
            // In FULL mode, apply a Y-axis correction to shift pitch center by -zeroOffsetDeg
            const float rad_off = (-sensors[i].zeroOffsetDeg) * ((float)M_PI / 180.0f);
            const float half_off = 0.5f * rad_off;
            FusionQuaternion qcorr;
            qcorr.element.w = cosf(half_off);
            qcorr.element.x = 0.0f;
            qcorr.element.y = sinf(half_off);
            qcorr.element.z = 0.0f;
            quat = FusionQuaternionMultiply(qcorr, quat);
        }
        sensors_data[i][0] = quat.element.w;
        sensors_data[i][1] = quat.element.x;
        sensors_data[i][2] = quat.element.y;
        sensors_data[i][3] = quat.element.z;
        sensors_data[i][4] = sensors[i].gyroscope_old.axis.x;
        sensors_data[i][5] = sensors[i].gyroscope_old.axis.y;
        sensors_data[i][6] = sensors[i].gyroscope_old.axis.z;
        // No pitch or raw accel in output CSV
    }
    // Use current time as stream timestamp (microseconds since boot)
    uint64_t out_ts = time_us_64();
    print_output_data(out_ts, (float (*)[11])sensors_data);
    // Check for runtime ZERO command
    check_for_zero_command();
    if (zero_requested) {
        // Re-zero support retained but not reflected in CSV output
        for (int i=0; i<imu_reader_settings.sensorCount; i++) {
            sensors[i].zeroOffsetDeg = sensors[i].pitchDeg;
        }
        zero_requested = 0;
        printf("Re-zero applied.\n");
    }
    uint64_t loop_end = time_us_64();
    uint64_t elapsed_us = loop_end - loop_start;
    uint64_t target_us = (uint64_t)(sleep_time * 1000.0f);
    if (elapsed_us < target_us) {
        sleep_us((uint32_t)(target_us - elapsed_us));
    }
    }
}

void update_loop_no_lpf(float sleep_time, float sensors_data[][11], Sensor* sensors) {
     while (true) {
    uint64_t loop_start = time_us_64();
    read_all_sensors(sensors);
    for (int i=0; i<imu_reader_settings.sensorCount;i++) {
        sensors[i].gyroscope = FusionCalibrationInertial(sensors[i].gyroscope, sensors[i].calibration.gyroscopeMisalignment, sensors[i].calibration.gyroscopeSensitivity, sensors[i].calibration.gyroscopeOffset);
        sensors[i].accelerometer = FusionCalibrationInertial(sensors[i].accelerometer, sensors[i].calibration.accelerometerMisalignment, sensors[i].calibration.accelerometerSensitivity, sensors[i].calibration.accelerometerOffset);
        sensors[i].gyroscope = FusionOffsetUpdate(&sensors[i].offset, sensors[i].gyroscope);

        const float deltaTime = (float) (sensors[i].timestamp - sensors[i].previousTimestamp) / 1e6f;
        sensors[i].previousTimestamp = sensors[i].timestamp;
        // FusionAhrsUpdateNoMagnetometer(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, deltaTime);

        FusionAhrsUpdateExternalHeading(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, 0.0f, deltaTime);
        FusionQuaternion quat = FusionAhrsGetQuaternion(&sensors[i].ahrs);
        quat = enforce_quaternion_continuity(quat, sensors[i].previousQuaternion);
        sensors[i].previousQuaternion = quat;  // Store for next iteration
        quat = apply_mounting(sensors[i].mountingQuaternion, quat);
        // Track pitch and optionally enforce pitch-only quaternion when requested
        float pitchDeg = quaternion_y_twist_deg(quat);
        float unwrapped = unwrap_angle_deg(pitchDeg, sensors[i].lastPitchDeg, &sensors[i].unwrapOffsetDeg);
        sensors[i].lastPitchDeg = pitchDeg;
        sensors[i].pitchDeg = unwrapped - sensors[i].zeroOffsetDeg;
        if (imu_reader_settings.pitchOnly) {
            const float rad = sensors[i].pitchDeg * ((float)M_PI / 180.0f);
            const float half = 0.5f * rad;
            const float c = cosf(half);
            const float s = sinf(half);
            quat.element.w = c;
            quat.element.x = 0.0f;
            quat.element.y = s;
            quat.element.z = 0.0f;
        } else if (sensors[i].zeroOffsetDeg != 0.0f) {
            // In FULL mode, apply a Y-axis correction to shift pitch center by -zeroOffsetDeg
            const float rad_off = (-sensors[i].zeroOffsetDeg) * ((float)M_PI / 180.0f);
            const float half_off = 0.5f * rad_off;
            FusionQuaternion qcorr;
            qcorr.element.w = cosf(half_off);
            qcorr.element.x = 0.0f;
            qcorr.element.y = sinf(half_off);
            qcorr.element.z = 0.0f;
            quat = FusionQuaternionMultiply(qcorr, quat);
        }
        sensors_data[i][0] = quat.element.w;
        sensors_data[i][1] = quat.element.x;
        sensors_data[i][2] = quat.element.y;
        sensors_data[i][3] = quat.element.z;
        sensors_data[i][4] = sensors[i].gyroscope.axis.x;
        sensors_data[i][5] = sensors[i].gyroscope.axis.y;
        sensors_data[i][6] = sensors[i].gyroscope.axis.z;
        // No pitch or raw accel in output CSV
    }
    uint64_t out_ts = time_us_64();
    print_output_data(out_ts, (float (*)[11])sensors_data);
    check_for_zero_command();
    if (zero_requested) {
        for (int i=0; i<imu_reader_settings.sensorCount; i++) {
            sensors[i].zeroOffsetDeg = sensors[i].pitchDeg;
        }
        zero_requested = 0;
        printf("Re-zero applied.\n");
    }
    uint64_t loop_end = time_us_64();
    uint64_t elapsed_us = loop_end - loop_start;
    uint64_t target_us = (uint64_t)(sleep_time * 1000.0f);
    if (elapsed_us < target_us) {
        sleep_us((uint32_t)(target_us - elapsed_us));
    }
    }   
}

static void startup_zero_alignment(Sensor* sensors) {
    const uint32_t duration_ms = 3000;
    const uint64_t start_us = time_us_64();
    float accumPitch[8] = {0};
    int counts[8] = {0};
    while ((time_us_64() - start_us) / 1000ULL < duration_ms) {
        read_all_sensors(sensors);
        for (int i=0; i<imu_reader_settings.sensorCount; i++) {
            sensors[i].gyroscope = FusionCalibrationInertial(sensors[i].gyroscope, sensors[i].calibration.gyroscopeMisalignment, sensors[i].calibration.gyroscopeSensitivity, sensors[i].calibration.gyroscopeOffset);
            sensors[i].accelerometer = FusionCalibrationInertial(sensors[i].accelerometer, sensors[i].calibration.accelerometerMisalignment, sensors[i].calibration.accelerometerSensitivity, sensors[i].calibration.accelerometerOffset);
            sensors[i].gyroscope = FusionOffsetUpdate(&sensors[i].offset, sensors[i].gyroscope);
            const float deltaTime = (float) (sensors[i].timestamp - sensors[i].previousTimestamp) / 1e6f;
            sensors[i].previousTimestamp = sensors[i].timestamp;
            FusionAhrsUpdateExternalHeading(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, 0.0f, deltaTime);
            FusionQuaternion q = FusionAhrsGetQuaternion(&sensors[i].ahrs);
            q = enforce_quaternion_continuity(q, sensors[i].previousQuaternion);
            sensors[i].previousQuaternion = q;  // Update for continuity during startup
            q = apply_mounting(sensors[i].mountingQuaternion, q);
            float pitch = quaternion_y_twist_deg(q);
            accumPitch[i] += pitch;
            counts[i]++;
        }
        sleep_ms(10);
    }
    for (int i=0; i<imu_reader_settings.sensorCount; i++) {
        if (counts[i] > 0) {
            sensors[i].zeroOffsetDeg = accumPitch[i] / (float)counts[i];
        } else {
            sensors[i].zeroOffsetDeg = 0.0f;
        }
    }
    printf("Startup zero complete.\n");
}

int main() {
    stdio_init_all();
    while (!tud_cdc_connected()) {
        sleep_ms(100);
    }
    wait_for_settings();
    // setup_sh2_service();
    
    float sensors_data[imu_reader_settings.sensorCount][11];
    float desired_loop_duration = 1000.0f/(float)imu_reader_settings.sampleRate;
    float sleep_time = desired_loop_duration; // pass desired period; loops will adjust
    int result = setup_I2C_pins();

    if (result != 1) {
		printf("I2C pin setup failed");
		return 1;
    }

	initialize_sensors();

    Sensor sensors[imu_reader_settings.sensorCount];
    initialize_sensors_values(sensors);
    initialize_calibrations(sensors); 
    initialize_algos(sensors);   

    // Default reference: world gravity (no auto-zero). Use CMD=ZERO to set custom offsets.

    if (imu_reader_settings.lpfEnabled == 1) {
        update_loop_with_lpf(sleep_time, sensors_data, sensors);
        printf("starting Update loop with lpf! \n");
    } else {
        update_loop_no_lpf(sleep_time, sensors_data, sensors);
        printf("starting Update loop no lpf! \n");
    }

    return 0;
}

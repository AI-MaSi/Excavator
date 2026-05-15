#include "i2c_helpers.h"
#include "InitFusion.h"
#include "Fusion.h"
#include "ism330dlc.h"
#include "tusb.h"
#include "cdc_console.h"
#include "status_led.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "settings.h"
#include "output.h"
#include <math.h>
#include <pico/multicore.h>
#include <pico/stdlib.h>
#include <string.h>

// MAX_SENSORS and FLOATS_PER_SENSOR defined in output.h

#define SENSOR_INTERNAL_ODR_HZ 416u
#define SENSOR_OUTPUT_HZ       200u

#define STARTUP_CALIBRATION_SETTLE_MS 3000u
#define STARTUP_CALIBRATION_DURATION_MS 30000u
#define STARTUP_CALIBRATION_SAMPLE_PERIOD_US 10000u
#define STARTUP_CALIBRATION_MIN_SAMPLES 1000u
#define CALIBRATION_MAX_GYRO_STD_DPS 0.5f
#define CALIBRATION_MAX_ACCEL_STD_G 0.05f
#define CALIBRATION_MIN_ACCEL_NORM_G 0.75f
#define CALIBRATION_MAX_ACCEL_NORM_G 1.25f

typedef struct imu_stream_slot_t {
    uint32_t sequence;
    uint32_t timestamp_us;
    uint8_t sensor_count;
    float sensors_data[MAX_SENSORS][FLOATS_PER_SENSOR];
} imu_stream_slot_t;

typedef struct usb_stream_bridge_t {
    volatile bool descriptor_ready;
    volatile bool stream_started;
    volatile uint16_t sample_rate_hz;
    volatile uint8_t sensor_count;
    uint8_t bus_ids[MAX_SENSORS];
    uint8_t sensor_addrs[MAX_SENSORS];
    imu_stream_slot_t slots[2];
    volatile uint32_t published_sequence;
    volatile uint8_t published_slot;
} usb_stream_bridge_t;

static usb_stream_bridge_t g_usb_stream = {0};

typedef struct sensor_runtime_t {
    float period_ms;
    uint8_t sensor_count;
    Sensor sensors[MAX_SENSORS];
} sensor_runtime_t;

typedef struct calibration_stats_t {
    FusionVector gyro_sum;
    FusionVector gyro_sum_sq;
    FusionVector accel_sum;
    FusionVector accel_sum_sq;
    float accel_norm_sum;
} calibration_stats_t;

static sensor_runtime_t g_sensor_runtime = {0};

static void update_loop(float period_ms, float sensors_data[][FLOATS_PER_SENSOR], Sensor* sensors, uint8_t sensor_count);

static inline FusionVector vector_add_square(FusionVector sum, FusionVector value) {
    sum.axis.x += value.axis.x * value.axis.x;
    sum.axis.y += value.axis.y * value.axis.y;
    sum.axis.z += value.axis.z * value.axis.z;
    return sum;
}

static inline FusionVector vector_divide(FusionVector vector, float scalar) {
    vector.axis.x /= scalar;
    vector.axis.y /= scalar;
    vector.axis.z /= scalar;
    return vector;
}

static inline FusionVector vector_std(FusionVector sum, FusionVector sum_sq, float count) {
    const FusionVector mean = vector_divide(sum, count);
    FusionVector variance = vector_divide(sum_sq, count);
    variance.axis.x -= mean.axis.x * mean.axis.x;
    variance.axis.y -= mean.axis.y * mean.axis.y;
    variance.axis.z -= mean.axis.z * mean.axis.z;

    FusionVector stddev;
    stddev.axis.x = sqrtf(fmaxf(variance.axis.x, 0.0f));
    stddev.axis.y = sqrtf(fmaxf(variance.axis.y, 0.0f));
    stddev.axis.z = sqrtf(fmaxf(variance.axis.z, 0.0f));
    return stddev;
}

static inline float vector_norm(FusionVector vector) {
    return sqrtf((vector.axis.x * vector.axis.x) +
                 (vector.axis.y * vector.axis.y) +
                 (vector.axis.z * vector.axis.z));
}

static inline calibration_vector_t calibration_vector_from_fusion(FusionVector vector) {
    return (calibration_vector_t){
        .x = vector.axis.x,
        .y = vector.axis.y,
        .z = vector.axis.z,
    };
}

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

static inline float tilt_pitch_deg_from_gravity(const FusionVector gravity) {
    const float gy = gravity.axis.y;
    const float gz = gravity.axis.z;
    return (180.0f / (float)M_PI) * atan2f(-gravity.axis.x, sqrtf(gy * gy + gz * gz));
}

static inline float tilt_roll_deg_from_gravity(const FusionVector gravity) {
    return (180.0f / (float)M_PI) * atan2f(gravity.axis.y, gravity.axis.z);
}

static inline bool fusion_vector_is_finite(const FusionVector vector) {
    return isfinite(vector.axis.x) && isfinite(vector.axis.y) && isfinite(vector.axis.z);
}

static inline bool fusion_quaternion_is_finite(const FusionQuaternion quaternion) {
    return isfinite(quaternion.element.w) && isfinite(quaternion.element.x) &&
           isfinite(quaternion.element.y) && isfinite(quaternion.element.z);
}

static inline float clamp_delta_time(float delta_time_s, float nominal_s) {
    if (!isfinite(delta_time_s) || (delta_time_s <= 0.0f)) {
        return nominal_s;
    }
    if (delta_time_s < nominal_s * 0.5f) {
        return nominal_s * 0.5f;
    }
    if (delta_time_s > nominal_s * 2.0f) {
        return nominal_s * 2.0f;
    }
    return delta_time_s;
}

static inline void write_sensor_output(float output[FLOATS_PER_SENSOR], FusionQuaternion quaternion, FusionVector gyroscope) {
    output[0] = quaternion.element.w;
    output[1] = quaternion.element.x;
    output[2] = quaternion.element.y;
    output[3] = quaternion.element.z;
    output[4] = gyroscope.axis.x;
    output[5] = gyroscope.axis.y;
    output[6] = gyroscope.axis.z;
}

static void calibration_service_until(uint64_t deadline_us) {
    static uint64_t next_cal_wait_us = 0;
    static uint64_t next_led_toggle_us = 0;
    static bool led_on = true;

    while (time_us_64() < deadline_us) {
        tud_task();
        const uint64_t now_us = time_us_64();

        if (now_us >= next_cal_wait_us) {
            send_control_msg(MSG_TYPE_CAL_WAIT);
            next_cal_wait_us = now_us + 200000u;
        }

        if (now_us >= next_led_toggle_us) {
            if (led_on) {
                status_led_set(STATUS_CALIBRATE);
            } else {
                status_led_off();
            }
            led_on = !led_on;
            next_led_toggle_us = now_us + 250000u;
        }

        const uint64_t remaining_us = deadline_us - now_us;
        if (remaining_us > 1000u) {
            sleep_us(500);
        } else {
            tight_loop_contents();
        }
    }
}

static void send_calibration_report_window(uint32_t duration_ms, uint8_t sensor_count, uint8_t flags,
                                           const calibration_report_sensor_t *reports) {
    const uint64_t start_us = time_us_64();
    uint64_t next_report_us = start_us;

    while ((time_us_64() - start_us) < 500000u) {
        tud_task();
        const uint64_t now_us = time_us_64();
        if (now_us >= next_report_us) {
            send_calibration_report(
                duration_ms,
                sensor_count,
                flags,
                CALIBRATION_MAX_GYRO_STD_DPS,
                CALIBRATION_MAX_ACCEL_STD_G,
                CALIBRATION_MIN_ACCEL_NORM_G,
                CALIBRATION_MAX_ACCEL_NORM_G,
                reports);
            next_report_us = now_us + 100000u;
        }
        sleep_ms(5);
    }
}

static bool startup_calibrate_gyro_biases(uint8_t sensor_count) {
    calibration_stats_t stats[MAX_SENSORS] = {0};
    calibration_report_sensor_t reports[MAX_SENSORS] = {0};

    // Let sensor ODR cycles fill output registers before collecting samples.
    status_led_set(STATUS_CALIBRATE);
    calibration_service_until(time_us_64() + ((uint64_t)STARTUP_CALIBRATION_SETTLE_MS * 1000u));

    const uint64_t calibration_end_us = time_us_64() + ((uint64_t)STARTUP_CALIBRATION_DURATION_MS * 1000u);
    uint64_t next_sample_us = time_us_64();
    while (time_us_64() < calibration_end_us) {
        const uint64_t now_us = time_us_64();
        if (now_us < next_sample_us) {
            calibration_service_until(next_sample_us);
            continue;
        }
        next_sample_us += STARTUP_CALIBRATION_SAMPLE_PERIOD_US;

        for (uint8_t i = 0; i < sensor_count; i++) {
            FusionVector accelerometer;
            FusionVector gyroscope;
            if (!read_active_sensor_motion_unbiased(i, &accelerometer, &gyroscope) ||
                !fusion_vector_is_finite(accelerometer) ||
                !fusion_vector_is_finite(gyroscope)) {
                reports[i].read_error_count++;
                continue;
            }

            const float accel_norm_sample = vector_norm(accelerometer);
            if (reports[i].sample_count == 0u) {
                reports[i].accel_norm_min = accel_norm_sample;
                reports[i].accel_norm_max = accel_norm_sample;
            } else {
                reports[i].accel_norm_min = fminf(reports[i].accel_norm_min, accel_norm_sample);
                reports[i].accel_norm_max = fmaxf(reports[i].accel_norm_max, accel_norm_sample);
            }
            reports[i].sample_count++;
            stats[i].gyro_sum = FusionVectorAdd(stats[i].gyro_sum, gyroscope);
            stats[i].gyro_sum_sq = vector_add_square(stats[i].gyro_sum_sq, gyroscope);
            stats[i].accel_sum = FusionVectorAdd(stats[i].accel_sum, accelerometer);
            stats[i].accel_sum_sq = vector_add_square(stats[i].accel_sum_sq, accelerometer);
            stats[i].accel_norm_sum += accel_norm_sample;
        }
    }

    bool accepted = true;

    for (uint8_t i = 0; i < sensor_count; i++) {
        calibration_report_sensor_t *report = &reports[i];
        if (report->sample_count < STARTUP_CALIBRATION_MIN_SAMPLES) {
            report->failure_flags |= CAL_FAIL_INSUFFICIENT_SAMPLES;
            accepted = false;
            continue;
        }

        const float count = (float)report->sample_count;
        const FusionVector gyro_mean = vector_divide(stats[i].gyro_sum, count);
        const FusionVector gyro_std = vector_std(stats[i].gyro_sum, stats[i].gyro_sum_sq, count);
        const FusionVector accel_mean = vector_divide(stats[i].accel_sum, count);
        const FusionVector accel_std = vector_std(stats[i].accel_sum, stats[i].accel_sum_sq, count);
        const float accel_norm = stats[i].accel_norm_sum / count;

        report->gyro_mean = calibration_vector_from_fusion(gyro_mean);
        report->gyro_std = calibration_vector_from_fusion(gyro_std);
        report->accel_mean = calibration_vector_from_fusion(accel_mean);
        report->accel_std = calibration_vector_from_fusion(accel_std);
        report->accel_norm_mean = accel_norm;

        if ((gyro_std.axis.x > CALIBRATION_MAX_GYRO_STD_DPS) ||
            (gyro_std.axis.y > CALIBRATION_MAX_GYRO_STD_DPS) ||
            (gyro_std.axis.z > CALIBRATION_MAX_GYRO_STD_DPS)) {
            report->failure_flags |= CAL_FAIL_GYRO_STD;
        }
        if ((accel_std.axis.x > CALIBRATION_MAX_ACCEL_STD_G) ||
            (accel_std.axis.y > CALIBRATION_MAX_ACCEL_STD_G) ||
            (accel_std.axis.z > CALIBRATION_MAX_ACCEL_STD_G)) {
            report->failure_flags |= CAL_FAIL_ACCEL_STD;
        }
        if ((accel_norm < CALIBRATION_MIN_ACCEL_NORM_G) ||
            (accel_norm > CALIBRATION_MAX_ACCEL_NORM_G)) {
            report->failure_flags |= CAL_FAIL_ACCEL_NORM;
        }

        if (report->failure_flags != 0u) {
            accepted = false;
        }
    }

    send_calibration_report_window(
        STARTUP_CALIBRATION_DURATION_MS,
        sensor_count,
        accepted ? CAL_REPORT_ACCEPTED : 0u,
        reports);

    // Always apply the measured gyro bias and proceed — the report is
    // informational. Threshold failures indicate noisy conditions but the
    // bias mean is still valid and better than nothing.
    for (uint8_t i = 0; i < sensor_count; i++) {
        FusionVector gyro_bias;
        gyro_bias.axis.x = reports[i].gyro_mean.x;
        gyro_bias.axis.y = reports[i].gyro_mean.y;
        gyro_bias.axis.z = reports[i].gyro_mean.z;
        set_active_sensor_gyro_bias(i, gyro_bias);
    }

    status_led_set(STATUS_INIT);
    return true;
}

static void calibration_error_forever(void) {
    bool led_on = true;
    uint64_t next_msg_us = 0;

    while (true) {
        tud_task();
        const uint64_t now_us = time_us_64();
        if (now_us >= next_msg_us) {
            send_control_msg(MSG_TYPE_ERR_CAL);
            next_msg_us = now_us + 200000u;
        }

        if (led_on) {
            status_led_set_rgb(24, 0, 0);
        } else {
            status_led_off();
        }
        led_on = !led_on;
        sleep_ms(150);
    }
}

static void stream_publish_descriptor(uint16_t sample_rate_hz, uint8_t sensor_count,
                                      const uint8_t *sensor_bus_ids, const uint8_t *sensor_addrs) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    memcpy((void *)g_usb_stream.bus_ids, sensor_bus_ids, sensor_count);
    memcpy((void *)g_usb_stream.sensor_addrs, sensor_addrs, sensor_count);
    g_usb_stream.sample_rate_hz = sample_rate_hz;
    g_usb_stream.sensor_count = sensor_count;
    __atomic_store_n(&g_usb_stream.descriptor_ready, true, __ATOMIC_RELEASE);
}

static void stream_publish_frame(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR], uint8_t sensor_count) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    const uint8_t next_slot = (uint8_t)((__atomic_load_n(&g_usb_stream.published_slot, __ATOMIC_RELAXED) + 1u) & 0x01u);
    imu_stream_slot_t *slot = &g_usb_stream.slots[next_slot];
    slot->timestamp_us = (uint32_t)ts_us;
    slot->sensor_count = sensor_count;
    memcpy(slot->sensors_data, sensors_data, sizeof(slot->sensors_data));

    const uint32_t next_sequence = __atomic_load_n(&g_usb_stream.published_sequence, __ATOMIC_RELAXED) + 1u;
    slot->sequence = next_sequence;
    __atomic_store_n(&g_usb_stream.published_slot, next_slot, __ATOMIC_RELEASE);
    __atomic_store_n(&g_usb_stream.published_sequence, next_sequence, __ATOMIC_RELEASE);
    __atomic_store_n(&g_usb_stream.stream_started, true, __ATOMIC_RELEASE);
}

static bool stream_copy_latest(imu_stream_slot_t *out, uint32_t *last_sequence) {
    while (true) {
        const uint32_t sequence = __atomic_load_n(&g_usb_stream.published_sequence, __ATOMIC_ACQUIRE);
        if ((sequence == 0u) || (sequence == *last_sequence)) {
            return false;
        }

        const uint8_t slot_index = __atomic_load_n(&g_usb_stream.published_slot, __ATOMIC_ACQUIRE);
        memcpy(out, &g_usb_stream.slots[slot_index], sizeof(*out));

        const uint32_t confirm_sequence = __atomic_load_n(&g_usb_stream.published_sequence, __ATOMIC_ACQUIRE);
        if ((confirm_sequence == sequence) && (out->sequence == sequence)) {
            *last_sequence = sequence;
            return true;
        }
    }
}

static void sleep_until_deadline(uint64_t deadline_us) {
    while (true) {
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

static void usb_transport_loop(void) {
    uint32_t last_sequence = 0;
    uint8_t descriptor_retries_remaining = 5;
    uint64_t next_descriptor_us = time_us_64();

    while (true) {
        tud_task();

        const uint64_t now_us = time_us_64();
        if (__atomic_load_n(&g_usb_stream.descriptor_ready, __ATOMIC_ACQUIRE) &&
            (descriptor_retries_remaining > 0u) &&
            (now_us >= next_descriptor_us)) {
            send_descriptor_frame(
                g_usb_stream.sample_rate_hz,
                g_usb_stream.sensor_count,
                g_usb_stream.bus_ids,
                g_usb_stream.sensor_addrs);
            descriptor_retries_remaining--;
            next_descriptor_us = now_us + 200000u;
        }

        imu_stream_slot_t latest;
        if (stream_copy_latest(&latest, &last_sequence)) {
            print_output_data(latest.timestamp_us, latest.sensors_data, latest.sensor_count);
            continue;
        }

        if (!__atomic_load_n(&g_usb_stream.stream_started, __ATOMIC_ACQUIRE)) {
            sleep_ms(1);
        } else {
            sleep_us(500);
        }
    }
}

static void sensor_core1_main(void) {
    float sensors_data[MAX_SENSORS][FLOATS_PER_SENSOR] = {{0.0f}};
    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        write_sensor_output(sensors_data[i], FUSION_IDENTITY_QUATERNION, FUSION_VECTOR_ZERO);
    }
    update_loop(g_sensor_runtime.period_ms, sensors_data, g_sensor_runtime.sensors, g_sensor_runtime.sensor_count);
}

// Main sensor loop - clean AHRS, no software filtering
static void update_loop(float period_ms, float sensors_data[][FLOATS_PER_SENSOR], Sensor* sensors, uint8_t sensor_count) {
    const uint64_t target_us = (uint64_t)(period_ms * 1000.0f + 0.5f);
    const float nominal_delta_time_s = period_ms / 1000.0f;
    const uint64_t output_period_us = 1000000u / SENSOR_OUTPUT_HZ;
    bool sensor_valid[MAX_SENSORS] = {false};
    uint64_t next_loop_us = time_us_64();
    uint64_t next_output_us = time_us_64();

    while (true) {
        read_all_sensors(sensors, sensor_valid);

        for (uint8_t i = 0; i < sensor_count; i++) {
            if (!sensor_valid[i] ||
                !fusion_vector_is_finite(sensors[i].accelerometer) ||
                !fusion_vector_is_finite(sensors[i].gyroscope)) {
                sensors[i].previousTimestamp = sensors[i].timestamp;
                write_sensor_output(sensors_data[i], sensors[i].previousQuaternion, FUSION_VECTOR_ZERO);
                continue;
            }

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

            if (!fusion_vector_is_finite(sensors[i].accelerometer) ||
                !fusion_vector_is_finite(sensors[i].gyroscope)) {
                sensors[i].previousTimestamp = sensors[i].timestamp;
                write_sensor_output(sensors_data[i], sensors[i].previousQuaternion, FUSION_VECTOR_ZERO);
                continue;
            }

            // Compute delta time
            const float deltaTime = clamp_delta_time(
                (float)(sensors[i].timestamp - sensors[i].previousTimestamp) / 1e6f,
                nominal_delta_time_s);
            sensors[i].previousTimestamp = sensors[i].timestamp;

            // AHRS update without magnetometer. Yaw is relative gyro integration;
            // do not force an external zero heading or slew motion is suppressed.
            FusionAhrsUpdate(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, FUSION_VECTOR_ZERO, deltaTime);

            // Get quaternion and enforce continuity
            FusionQuaternion quat = FusionAhrsGetQuaternion(&sensors[i].ahrs);
            if (!fusion_quaternion_is_finite(quat)) {
                FusionAhrsReset(&sensors[i].ahrs);
                quat = FUSION_IDENTITY_QUATERNION;
            }
            quat = enforce_quaternion_continuity(quat, sensors[i].previousQuaternion);
            sensors[i].previousQuaternion = quat;

            // Derive tilt from gravity so pitch/roll stay stable under yaw motion
            const FusionVector gravity = FusionAhrsGetGravity(&sensors[i].ahrs);
            sensors[i].pitchDeg = tilt_pitch_deg_from_gravity(gravity);
            sensors[i].rollDeg = tilt_roll_deg_from_gravity(gravity);

            write_sensor_output(sensors_data[i], quat, sensors[i].gyroscope);
        }

        const uint64_t now_us = time_us_64();
        if (now_us >= next_output_us) {
            stream_publish_frame(now_us, sensors_data, sensor_count);
            next_output_us += output_period_us;
            if (now_us >= next_output_us)
                next_output_us = now_us + output_period_us;
        }

        // Maintain a stable absolute deadline instead of sleeping relative to
        // the loop start each iteration.
        next_loop_us += target_us;
        uint64_t schedule_now_us = time_us_64();

        if (schedule_now_us < next_loop_us) {
            sleep_until_deadline(next_loop_us);
        } else if ((schedule_now_us - next_loop_us) > target_us) {
            // If we fall badly behind, re-anchor to the current time rather
            // than carrying the slip forever.
            next_loop_us = schedule_now_us;
        }
    }
}

int main() {
    // Init TinyUSB directly — do NOT call stdio_init_all() as the
    // stdio USB driver steals bytes from our tud_cdc_read() calls.
    tusb_init();

    // Init NeoPixel status LED first (no USB needed)
    status_led_init();
    status_led_set(STATUS_BOOT);  // Amber — waiting for USB

    while (!tud_cdc_connected()) {
        status_led_set(STATUS_BOOT);
        tud_task();
        sleep_ms(10);
    }

    // Setup I2C after CDC is connected so fatal init errors reach the host.
    if (!setup_I2C_pins(400*1000)) {
        status_led_set(STATUS_ERROR);
        send_control_msg(MSG_TYPE_ERR_I2C);
        while (1) { tud_task(); sleep_ms(100); }
    }

    status_led_set(STATUS_INIT);

    if (!initialize_sensors(SENSOR_INTERNAL_ODR_HZ)) {
        status_led_set(STATUS_ERROR);
        send_control_msg(MSG_TYPE_ERR_IMU);
        while (1) { tud_task(); sleep_ms(100); }
    }

    uint8_t sensor_count = get_active_sensor_count();
    if (sensor_count > MAX_SENSORS) sensor_count = MAX_SENSORS;
    uint8_t sensor_bus_ids[MAX_SENSORS];
    uint8_t sensor_addrs[MAX_SENSORS];
    for (uint8_t i = 0; i < sensor_count; i++) {
        sensor_bus_ids[i] = get_active_sensor_bus(i);
        sensor_addrs[i] = get_active_sensor_addr(i);
    }

    if (!startup_calibrate_gyro_biases(sensor_count)) {
        calibration_error_forever();
    }

    g_sensor_runtime.sensor_count = sensor_count;
    g_sensor_runtime.period_ms = 1000.0f / (float)SENSOR_INTERNAL_ODR_HZ;
    initialize_sensors_values(g_sensor_runtime.sensors, MAX_SENSORS);
    initialize_calibrations(g_sensor_runtime.sensors, MAX_SENSORS);
    initialize_algos(g_sensor_runtime.sensors, MAX_SENSORS);

    stream_publish_descriptor(SENSOR_OUTPUT_HZ, sensor_count, sensor_bus_ids, sensor_addrs);
    status_led_set(STATUS_STREAM);
    multicore_launch_core1(sensor_core1_main);
    usb_transport_loop();

    return 0;
}

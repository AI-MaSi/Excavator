#include "output.h"
#include "tusb.h"
#include <pico/stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define FRAME_SYNC_0 0xAA
#define FRAME_SYNC_1 0x55
#define FRAME_TYPE_DATA 1
#define FRAME_TYPE_CTRL 2
#define FRAME_TYPE_DESC 3
#define FRAME_TYPE_CAL_REPORT 5

static bool cdc_send_frame_best_effort(const uint8_t *frame, size_t len, bool drop_if_full) {
    if (!tud_cdc_connected()) {
        return false;
    }

    if (drop_if_full) {
        if (tud_cdc_write_available() < len) {
            tud_cdc_write_flush();
            return false;
        }

        const uint32_t written = tud_cdc_write(frame, len);
        if (written != len) {
            tud_cdc_write_flush();
            return false;
        }

        tud_cdc_write_flush();
        return true;
    }

    const uint64_t start_us = time_us_64();
    size_t written_total = 0;
    while (written_total < len) {
        tud_task();
        const uint32_t available = tud_cdc_write_available();
        if (available == 0u) {
            tud_cdc_write_flush();
            if ((time_us_64() - start_us) > 500000u) {
                return false;
            }
            sleep_us(100);
            continue;
        }

        const size_t remaining = len - written_total;
        const size_t chunk_len = (available < remaining) ? available : remaining;
        const uint32_t written = tud_cdc_write(frame + written_total, chunk_len);
        if (written == 0u) {
            if ((time_us_64() - start_us) > 500000u) {
                return false;
            }
            sleep_us(100);
            continue;
        }

        written_total += written;
        tud_cdc_write_flush();
    }

    return true;
}

// Binary data frame:
// [0xAA 0x55] [frame_type:1] [count:1] [timestamp_us:4] [sensor0..N: 7 floats each] [checksum:2]
// Each sensor: w, x, y, z (quaternion), gx, gy, gz (gyro dps)

void print_output_data(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR], uint8_t sensor_count) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    // Buffer sized for the maximum supported sensor count.
    uint8_t frame[4 + 4 + (MAX_SENSORS * FLOATS_PER_SENSOR * 4) + 2];
    size_t idx = 0;

    // Sync + header
    frame[idx++] = FRAME_SYNC_0;
    frame[idx++] = FRAME_SYNC_1;
    frame[idx++] = FRAME_TYPE_DATA;
    frame[idx++] = sensor_count;

    // Timestamp (truncated to 32-bit, wraps every ~71 minutes)
    uint32_t ts32 = (uint32_t)ts_us;
    memcpy(&frame[idx], &ts32, sizeof(ts32));
    idx += sizeof(ts32);

    // Sensor data
    for (uint8_t i = 0; i < sensor_count; i++) {
        for (int j = 0; j < FLOATS_PER_SENSOR; j++) {
            float v = sensors_data[i][j];
            memcpy(&frame[idx], &v, sizeof(float));
            idx += sizeof(float);
        }
    }

    // Simple checksum (sum of bytes from version to end of payload)
    uint16_t checksum = 0;
    for (size_t i = 2; i < idx; i++) {
        checksum = (uint16_t)(checksum + frame[i]);
    }
    frame[idx++] = (uint8_t)(checksum & 0xFF);
    frame[idx++] = (uint8_t)((checksum >> 8) & 0xFF);

    (void)cdc_send_frame_best_effort(frame, idx, true);
}

// Send a control message (CFG_OK, CFG_WAIT, etc.)
// Frame: [0xAA 0x55] [frame_type=2] [msg_type] [checksum:2]
void send_control_msg(uint8_t msg_type) {
    uint8_t frame[6];
    frame[0] = FRAME_SYNC_0;
    frame[1] = FRAME_SYNC_1;
    frame[2] = FRAME_TYPE_CTRL;
    frame[3] = msg_type;

    uint16_t checksum = frame[2] + frame[3];
    frame[4] = (uint8_t)(checksum & 0xFF);
    frame[5] = (uint8_t)((checksum >> 8) & 0xFF);

    (void)cdc_send_frame_best_effort(frame, sizeof(frame), false);
}

// Send a one-shot descriptor frame so the host can map stream index -> bus/address.
// Frame: [0xAA 0x55] [frame_type=3] [count:1] [sample_rate_hz:2] [bus,addr]*N [checksum:2]
void send_descriptor_frame(uint16_t sample_rate_hz, uint8_t sensor_count, const uint8_t *bus_ids, const uint8_t *device_addrs) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    uint8_t frame[4 + 2 + (MAX_SENSORS * 2) + 2];
    size_t idx = 0;

    frame[idx++] = FRAME_SYNC_0;
    frame[idx++] = FRAME_SYNC_1;
    frame[idx++] = FRAME_TYPE_DESC;
    frame[idx++] = sensor_count;
    memcpy(&frame[idx], &sample_rate_hz, sizeof(sample_rate_hz));
    idx += sizeof(sample_rate_hz);

    for (uint8_t i = 0; i < sensor_count; i++) {
        frame[idx++] = bus_ids[i];
        frame[idx++] = device_addrs[i];
    }

    uint16_t checksum = 0;
    for (size_t i = 2; i < idx; i++) {
        checksum = (uint16_t)(checksum + frame[i]);
    }
    frame[idx++] = (uint8_t)(checksum & 0xFF);
    frame[idx++] = (uint8_t)((checksum >> 8) & 0xFF);

    (void)cdc_send_frame_best_effort(frame, idx, false);
}

// Send one startup calibration report before CFG_OK or ERR_CAL.
// Frame: [0xAA 0x55] [frame_type=5] [count:1] [duration_ms:4]
//        [flags:1] [reserved:1] [thresholds:4 floats] [per-sensor stats] [checksum:2]
void send_calibration_report(uint32_t duration_ms, uint8_t sensor_count, uint8_t flags,
                             float gyro_std_limit_dps, float accel_std_limit_g,
                             float accel_norm_min_g, float accel_norm_max_g,
                             const calibration_report_sensor_t *sensor_reports) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    uint8_t frame[4 + 4 + 2 + (4 * sizeof(float)) + (MAX_SENSORS * 72) + 2];
    size_t idx = 0;

    frame[idx++] = FRAME_SYNC_0;
    frame[idx++] = FRAME_SYNC_1;
    frame[idx++] = FRAME_TYPE_CAL_REPORT;
    frame[idx++] = sensor_count;

    memcpy(&frame[idx], &duration_ms, sizeof(duration_ms));
    idx += sizeof(duration_ms);
    frame[idx++] = flags;
    frame[idx++] = 0u;

    memcpy(&frame[idx], &gyro_std_limit_dps, sizeof(float));
    idx += sizeof(float);
    memcpy(&frame[idx], &accel_std_limit_g, sizeof(float));
    idx += sizeof(float);
    memcpy(&frame[idx], &accel_norm_min_g, sizeof(float));
    idx += sizeof(float);
    memcpy(&frame[idx], &accel_norm_max_g, sizeof(float));
    idx += sizeof(float);

    for (uint8_t i = 0; i < sensor_count; i++) {
        const calibration_report_sensor_t *report = &sensor_reports[i];
        const uint16_t reserved = 0u;

        memcpy(&frame[idx], &report->sample_count, sizeof(report->sample_count));
        idx += sizeof(report->sample_count);
        memcpy(&frame[idx], &report->read_error_count, sizeof(report->read_error_count));
        idx += sizeof(report->read_error_count);
        memcpy(&frame[idx], &report->failure_flags, sizeof(report->failure_flags));
        idx += sizeof(report->failure_flags);
        memcpy(&frame[idx], &reserved, sizeof(reserved));
        idx += sizeof(reserved);

        memcpy(&frame[idx], &report->gyro_mean.x, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->gyro_mean.y, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->gyro_mean.z, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->gyro_std.x, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->gyro_std.y, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->gyro_std.z, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_mean.x, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_mean.y, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_mean.z, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_std.x, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_std.y, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_std.z, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_norm_mean, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_norm_min, sizeof(float));
        idx += sizeof(float);
        memcpy(&frame[idx], &report->accel_norm_max, sizeof(float));
        idx += sizeof(float);
    }

    uint16_t checksum = 0;
    for (size_t i = 2; i < idx; i++) {
        checksum = (uint16_t)(checksum + frame[i]);
    }
    frame[idx++] = (uint8_t)(checksum & 0xFF);
    frame[idx++] = (uint8_t)((checksum >> 8) & 0xFF);

    (void)cdc_send_frame_best_effort(frame, idx, false);
}

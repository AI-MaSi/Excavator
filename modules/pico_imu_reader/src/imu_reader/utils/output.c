#include "output.h"
#include "tusb.h"
#include <string.h>

#define FRAME_SYNC_0 0xAA
#define FRAME_SYNC_1 0x55
#define FRAME_VERSION_DATA 1
#define FRAME_VERSION_CTRL 2
#define NUM_SENSORS 3
#define FLOATS_PER_SENSOR 7

// Binary frame format:
// [0xAA 0x55] [version:1] [count:1] [timestamp_us:4] [sensor0..N: 7 floats each] [checksum:2]
// Each sensor: w, x, y, z (quaternion), gx, gy, gz (gyro dps)

void print_output_data(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR]) {
    if (!tud_cdc_connected()) {
        return;
    }

    // Frame buffer: header(4) + timestamp(4) + payload(3*7*4=84) + checksum(2) = 94 bytes
    uint8_t frame[4 + 4 + (NUM_SENSORS * FLOATS_PER_SENSOR * 4) + 2];
    size_t idx = 0;

    // Sync + header
    frame[idx++] = FRAME_SYNC_0;
    frame[idx++] = FRAME_SYNC_1;
    frame[idx++] = FRAME_VERSION_DATA;
    frame[idx++] = NUM_SENSORS;

    // Timestamp (truncated to 32-bit, wraps every ~71 minutes)
    uint32_t ts32 = (uint32_t)ts_us;
    memcpy(&frame[idx], &ts32, sizeof(ts32));
    idx += sizeof(ts32);

    // Sensor data
    for (int i = 0; i < NUM_SENSORS; i++) {
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

    // Wait for buffer space to avoid dropping bytes (with timeout)
    uint32_t avail = tud_cdc_write_available();
    if (avail < idx) {
        // Flush any pending data and wait briefly for space
        tud_cdc_write_flush();
        for (int retry = 0; retry < 10 && tud_cdc_write_available() < idx; retry++) {
            tud_task();  // Process USB events
        }
    }

    // Write entire frame atomically if possible
    uint32_t written = tud_cdc_write(frame, idx);
    if (written < idx) {
        // Buffer still full - write remaining bytes
        tud_cdc_write_flush();
        tud_cdc_write(frame + written, idx - written);
    }
    tud_cdc_write_flush();
}

// Send a control message (CFG_OK, ZERO_ACK, etc.)
// Frame: [0xAA 0x55] [version=2] [msg_type] [checksum:2]
void send_control_msg(uint8_t msg_type) {
    if (!tud_cdc_connected()) {
        return;
    }

    uint8_t frame[6];
    frame[0] = FRAME_SYNC_0;
    frame[1] = FRAME_SYNC_1;
    frame[2] = FRAME_VERSION_CTRL;
    frame[3] = msg_type;

    uint16_t checksum = frame[2] + frame[3];
    frame[4] = (uint8_t)(checksum & 0xFF);
    frame[5] = (uint8_t)((checksum >> 8) & 0xFF);

    tud_cdc_write(frame, sizeof(frame));
    tud_cdc_write_flush();
}

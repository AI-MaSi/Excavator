#include "settings.h"
#include "cdc_console.h"
#include "output.h"
#include "tusb.h"
#include <pico/stdlib.h>

volatile int zero_requested = 0;

// Default settings
imu_reader_settings_t imu_reader_settings = {
    .sampleRate = 200,
    .gyroRangeDps = 500.0f,
    .ahrsGain = 0.5f,
    .ahrsAccelRejection = 10.0f,
    .ahrsRecoveryPeriodS = 1.0f,
    .offsetTimeoutS = 1.0f
};
settings_enum settings_option;

static void extract_part(const char* key, const char* buf, settings_enum setting) {
    char *pos = strstr(buf, key);
    if (pos == NULL) return;  // Optional - skip if not found

    char part_buffer[64];
    memset(part_buffer, 0, sizeof(part_buffer));
    int cursor = 0;
    int part_index = (pos - buf) + strlen(key);

    for (int j = part_index; j < SETTINGS_BUF_LEN - part_index && cursor < 63; j++) {
        if (buf[j] == '|' || buf[j] == '\n' || buf[j] == '\0') break;
        part_buffer[cursor++] = buf[j];
    }
    part_buffer[cursor] = '\0';

    switch (setting) {
        case S_SAMPLE_RATE:
            imu_reader_settings.sampleRate = atoi(part_buffer);
            break;
        case S_GYRO_RANGE:
            sscanf(part_buffer, "%f", &imu_reader_settings.gyroRangeDps);
            break;
        case S_AHRS_GAIN:
            sscanf(part_buffer, "%f", &imu_reader_settings.ahrsGain);
            break;
        case S_AHRS_ACCEL_REJ:
            sscanf(part_buffer, "%f", &imu_reader_settings.ahrsAccelRejection);
            break;
        case S_AHRS_RECOVERY_S:
            sscanf(part_buffer, "%f", &imu_reader_settings.ahrsRecoveryPeriodS);
            break;
        case S_OFFSET_TIMEOUT_S:
            sscanf(part_buffer, "%f", &imu_reader_settings.offsetTimeoutS);
            break;
    }
    cdc_writef("Config: %s%s\n", key, part_buffer);
}

static void parse_settings(const char* buf) {
    // Required: sample rate
    extract_part("SR=", buf, S_SAMPLE_RATE);

    // Optional AHRS parameters (use defaults if not provided)
    extract_part("GYRO_DPS=", buf, S_GYRO_RANGE);
    extract_part("GAIN=", buf, S_AHRS_GAIN);
    extract_part("ACC_REJ=", buf, S_AHRS_ACCEL_REJ);
    extract_part("RECOV_S=", buf, S_AHRS_RECOVERY_S);
    extract_part("OFFSET_S=", buf, S_OFFSET_TIMEOUT_S);

    // Validate
    if (imu_reader_settings.sampleRate < 10) imu_reader_settings.sampleRate = 200;
    if (imu_reader_settings.sampleRate > 1000) imu_reader_settings.sampleRate = 1000;
    if (imu_reader_settings.gyroRangeDps < 125.0f) imu_reader_settings.gyroRangeDps = 500.0f;
    if (imu_reader_settings.ahrsGain <= 0.0f) imu_reader_settings.ahrsGain = 0.5f;

    cdc_writef("Settings: SR=%d Hz, GYRO=%.0f dps, GAIN=%.2f, ACC_REJ=%.1f, RECOV=%.2fs, OFFSET=%.2fs\n",
           imu_reader_settings.sampleRate,
           imu_reader_settings.gyroRangeDps,
           imu_reader_settings.ahrsGain,
           imu_reader_settings.ahrsAccelRejection,
           imu_reader_settings.ahrsRecoveryPeriodS,
           imu_reader_settings.offsetTimeoutS);
}

void wait_for_settings() {
    char buf[SETTINGS_BUF_LEN];
    cdc_write_line("Waiting for config (SR=xxx|...)...");

    absolute_time_t last_beat = get_absolute_time();

    while (1) {
        tud_task();
        if (absolute_time_diff_us(last_beat, get_absolute_time()) >= 1000 * 1000) {
            send_control_msg(MSG_TYPE_CFG_WAIT);
            last_beat = get_absolute_time();
        }

        if (tud_cdc_available()) {
            uint32_t count = tud_cdc_read(buf, sizeof(buf) - 1);
            if (count > 0) {
                buf[count] = '\0';

                // Handle ZERO command
                if (strstr(buf, "CMD=ZERO") != NULL) {
                    zero_requested = 1;
                    cdc_write_line("ZERO queued.");
                }

                // Basic connectivity test
                if (strstr(buf, "TEST") != NULL) {
                    cdc_write_line("TEST_OK");
                }

                // Parse settings if SR= is present
                if (strstr(buf, "SR=")) {
                    parse_settings(buf);
                    // Disable text console before streaming binary data
                    cdc_console_enable(false);
                    // Send CFG_OK for a short window, then exit to streaming.
                    absolute_time_t cfg_ok_start = get_absolute_time();
                    absolute_time_t last_cfg_ok = cfg_ok_start;
                    while (absolute_time_diff_us(cfg_ok_start, get_absolute_time()) < 500 * 1000) {
                        tud_task();
                        if (absolute_time_diff_us(last_cfg_ok, get_absolute_time()) >= 100 * 1000) {
                            send_control_msg(MSG_TYPE_CFG_OK);
                            last_cfg_ok = get_absolute_time();
                        }
                        sleep_ms(10);
                    }
                    return;
                }
            }
        }
        sleep_ms(100);
    }
}

void check_for_zero_command() {
    if (!tud_cdc_available()) return;

    char buf[SETTINGS_BUF_LEN];
    uint32_t count = tud_cdc_read(buf, sizeof(buf) - 1);
    if (count == 0) return;

    buf[count] = '\0';
    if (strstr(buf, "CMD=ZERO") != NULL) {
        zero_requested = 1;
        send_control_msg(MSG_TYPE_ZERO_ACK);
    }
}

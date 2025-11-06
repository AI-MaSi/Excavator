#include "settings.h"
#include "tusb.h"

volatile int zero_requested = 0;

void extract_part(const char* part, const char* buf, settings_enum setting) {
        char *pos = strstr(buf, part);
        
        if (pos == 0) {
            printf("Error while extracting part %s not found in the buffer\n", part);
            while(1);
        } else {
            char part_buffer[64];//SR=jotain|
            memset(part_buffer, 0, sizeof(part_buffer));
            int cursor = 0;
            int part_index = (pos - buf) + strlen(part);
            for (int j=part_index;j<SETTINGS_BUF_LEN-part_index; j++) {
                if (buf[j] != '|') {
                    part_buffer[cursor++] = buf[j];
                    continue;
                }
                part_buffer[cursor++] = '\0';
                break;
            }
            switch (setting) {
                case S_SENSOR_COUNT:
                    imu_reader_settings.sensorCount = atoi(part_buffer);
                    break;
                case S_LPF_ENABLED:
                    imu_reader_settings.lpfEnabled = atoi(part_buffer);
                    break;
                case S_LPF_ALPHA:
                    sscanf(part_buffer, "%f", &imu_reader_settings.lpf_alpha);
                    break;
                case S_SAMPLE_RATE:
                    imu_reader_settings.sampleRate = atoi(part_buffer);
                    break;
                case S_QMODE: {
                    // Accept PITCH/FULL (case-insensitive) or 1/0
                    if (part_buffer[0] == 'P' || part_buffer[0] == 'p' || strcmp(part_buffer, "1") == 0) {
                        imu_reader_settings.pitchOnly = 1;
                    } else {
                        imu_reader_settings.pitchOnly = 0;
                    }
                    break;
                }
                case S_STREAM_RAW:
                    imu_reader_settings.streamRaw = atoi(part_buffer);
                    break;
                case S_FORMAT: {
                    if (part_buffer[0] == 'C' || part_buffer[0] == 'c') {
                        imu_reader_settings.formatCSV = 1;
                    } else {
                        imu_reader_settings.formatCSV = 0;
                    }
                    break;
                }
                default:
                    printf("Error: Wrong setting %d\n", setting);
                    while(1);
            }
            printf("Part %s found and saved: %s\n", part, part_buffer);
        }
}

void excract_settings(const char* buf) {
    // Required: SC, SR, QMODE
    settings_option = S_SENSOR_COUNT;
    extract_part("SC=", buf, settings_option);

    settings_option = S_SAMPLE_RATE;
    extract_part("SR=", buf, settings_option);
    
    settings_option = S_QMODE;
    extract_part("QMODE=", buf, settings_option);

    // Optional LPF parameters (default off when not provided)
    if (strstr(buf, "LPF_ENABLED=") != NULL) {
        settings_option = S_LPF_ENABLED;
        extract_part("LPF_ENABLED=", buf, settings_option);
    }
    if (strstr(buf, "LPF_ALPHA=") != NULL) {
        settings_option = S_LPF_ALPHA;
        extract_part("LPF_ALPHA=", buf, settings_option);
    }

    printf("Settings: SC=%d; SR=%d; QMODE(pitchOnly)=%d; LPF=%d (alpha=%.3f)\n",
           imu_reader_settings.sensorCount,
           imu_reader_settings.sampleRate,
           imu_reader_settings.pitchOnly,
           imu_reader_settings.lpfEnabled,
           imu_reader_settings.lpf_alpha);
    (imu_reader_settings.sensorCount > 2) ? (imu_reader_settings.channelCount = 2) : (imu_reader_settings.channelCount = 1);
}

// Waits for the user to send in some settings through the serial port
void wait_for_settings() {

    char buf[SETTINGS_BUF_LEN];
    while (1) {
        if (tud_cdc_available()) {
            uint32_t count = tud_cdc_read(buf, sizeof(buf)-1);
            if (count > 0) {
                buf[count] = '\0';
                printf("Received settings: %s\n", buf);
                // Handle ZERO command if present
                if (strstr(buf, "CMD=ZERO") != NULL) {
                    zero_requested = 1;
                    printf("ZERO command queued.\n");
                }
                // Extract settings if at least one of the required keys is present
                if (strstr(buf, "SC=") || strstr(buf, "SR=") || strstr(buf, "QMODE=")) {
                    excract_settings(buf);
                    break;
                }
            }
        }
        sleep_ms(300); // Avoid tight loop
    }
}

// Non-blocking check for ZERO command during runtime
void check_for_zero_command() {
    if (!tud_cdc_available()) {
        return;
    }
    char buf[SETTINGS_BUF_LEN];
    uint32_t count = tud_cdc_read(buf, sizeof(buf)-1);
    if (count == 0) {
        return;
    }
    buf[count] = '\0';
    if (strstr(buf, "CMD=ZERO") != NULL) {
        zero_requested = 1;
        printf("ZERO command received. Will re-zero now.\n");
    }
}

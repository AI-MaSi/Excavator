#include "data_transfer.h"
#include "kalman_processing.h"
#include "imu_setup.h"

// Calculate checksum for binary packet
uint8_t calculate_checksum(const uint8_t* data, size_t length) {
    uint8_t checksum = 0;
    
    for (size_t i = 0; i < length; i++) {
        checksum ^= data[i];
    }
    
    return checksum;
}

// Send quaternion data via serial in binary format
void send_quaternion_data() {
    uint64_t current_time = time_us_64() / 1000; // ms
    
    // Only send data at the specified interval
    if (current_time - last_data_time < data_interval_ms && last_data_time > 0) {
        return;
    }
    
    last_data_time = current_time;
    
    // Prepare binary data packet
    uint8_t packet[128]; // Buffer for binary data
    int offset = 0;
    
    // Header
    packet[offset++] = DATA_PACKET_START_1;  // 0xAA
    packet[offset++] = DATA_PACKET_START_2;  // 0xBB
    
    // Packet counter
    packet_count++;
    packet[offset++] = (packet_count & 0xFF);
    
    // Number of IMUs
    packet[offset++] = imu_count;
    
    // Timestamp (32-bit, little-endian)
    uint32_t timestamp = (uint32_t)current_time;
    packet[offset++] = (timestamp & 0xFF);
    packet[offset++] = ((timestamp >> 8) & 0xFF);
    packet[offset++] = ((timestamp >> 16) & 0xFF);
    packet[offset++] = ((timestamp >> 24) & 0xFF);
    
    // Mark start of payload for checksum
    int payload_start = offset;
    
    // Add data for each IMU
    for (int i = 0; i < imu_count; i++) {
        // Verify quaternion before sending
        // Calculate magnitude
        float qw = imus[i].quaternion[0];
        float qx = imus[i].quaternion[1];
        float qy = imus[i].quaternion[2];
        float qz = imus[i].quaternion[3];
        
        float mag = qw*qw + qx*qx + qy*qy + qz*qz;
        
        // If magnitude is not close to 1 or any component is invalid, reset to identity
        if (mag < 0.95f || mag > 1.05f || !isfinite(qw) || !isfinite(qx) || 
            !isfinite(qy) || !isfinite(qz)) {
            imus[i].quaternion[0] = 1.0f;
            imus[i].quaternion[1] = 0.0f;
            imus[i].quaternion[2] = 0.0f;
            imus[i].quaternion[3] = 0.0f;
        }
        
        // IMU index
        packet[offset++] = (uint8_t)i;
        
        // Quaternion data (4 floats)
        float quat_values[4] = {
            imus[i].quaternion[0],
            imus[i].quaternion[1],
            imus[i].quaternion[2],
            imus[i].quaternion[3]
        };
        
        // Copy each quaternion component (4 floats, 16 bytes total)
        for (int j = 0; j < 4; j++) {
            uint8_t* bytes = (uint8_t*)&quat_values[j];
            packet[offset++] = bytes[0];
            packet[offset++] = bytes[1];
            packet[offset++] = bytes[2];
            packet[offset++] = bytes[3];
        }
        
        // Add verbose flag and accel/gyro data if enabled
        if (verbose_output) {
            packet[offset++] = 1; // Verbose flag = true
            
            // Add accelerometer data (3 floats)
            float accel_values[3] = {
                imus[i].accel[0],
                imus[i].accel[1],
                imus[i].accel[2]
            };
            
            for (int j = 0; j < 3; j++) {
                uint8_t* bytes = (uint8_t*)&accel_values[j];
                packet[offset++] = bytes[0];
                packet[offset++] = bytes[1];
                packet[offset++] = bytes[2];
                packet[offset++] = bytes[3];
            }
            
            // Add gyroscope data (3 floats)
            float gyro_values[3] = {
                imus[i].gyro[0],
                imus[i].gyro[1],
                imus[i].gyro[2]
            };
            
            for (int j = 0; j < 3; j++) {
                uint8_t* bytes = (uint8_t*)&gyro_values[j];
                packet[offset++] = bytes[0];
                packet[offset++] = bytes[1];
                packet[offset++] = bytes[2];
                packet[offset++] = bytes[3];
            }
        } else {
            packet[offset++] = 0; // Verbose flag = false
        }
    }
    
    // Calculate checksum over payload portion
    uint8_t checksum = calculate_checksum(&packet[payload_start], offset - payload_start);
    packet[offset++] = checksum;
    
    // End markers
    packet[offset++] = DATA_PACKET_END_1;  // 0xCC
    packet[offset++] = DATA_PACKET_END_2;  // 0xDD
    
    // Send the binary packet
    for (int i = 0; i < offset; i++) {
        putchar(packet[i]);
    }
    fflush(stdout);
}

// Process a binary command
void process_binary_command(uint8_t cmd_code, const uint8_t* data, uint8_t data_length) {
    if (debug_mode) {
        printf("Processing binary command: 0x%02X\n", cmd_code);
    }
    
    switch (cmd_code) {
        case CMD_START:
            // Make sure IMUs are initialized
            if (imu_count == 0) {
                imus[0].active = true;
                imus[0].channel = 0;
                imus[0].address = ISM330_ADDR_PRIMARY;
                imu_count = 1;
                
                init_kalman_filter(0);
            }
            
            // Clear any pending output before starting binary mode
            fflush(stdout);
            
            // Start sending data
            running = true;
            if (debug_mode) {
                printf("Started data streaming\n");
            }
            break;
            
        case CMD_STOP:
            running = false;
            if (debug_mode) {
                printf("Stopped data streaming\n");
            }
            break;
            
        case CMD_SCAN:
            scan_for_imus();
            
            // Configure all detected IMUs
            for (int i = 0; i < imu_count; i++) {
                configure_imu(i);
            }
            
            if (debug_mode) {
                printf("Scan completed, found %d IMUs\n", imu_count);
            }
            break;
            
        case CMD_STATUS:
            if (debug_mode) {
                printf("Status:\n");
                printf("  Running: %s\n", running ? "Yes" : "No");
                printf("  IMUs detected: %d\n", imu_count);
                printf("  Data update rate: %llu ms (%.1f Hz)\n", 
                       data_interval_ms, 1000.0f / data_interval_ms);
                
                for (int i = 0; i < imu_count; i++) {
                    printf("  IMU%d: Channel %d, Address 0x%02X\n", 
                           i, imus[i].channel, imus[i].address);
                    printf("    Quaternion: [%.4f, %.4f, %.4f, %.4f]\n", 
                           imus[i].quaternion[0], imus[i].quaternion[1], 
                           imus[i].quaternion[2], imus[i].quaternion[3]);
                    printf("    Gyro bias: [%.4f, %.4f, %.4f]\n",
                           imus[i].x_data[4], imus[i].x_data[5], imus[i].x_data[6]);
                }
            }
            break;
            
        case CMD_RESETBIAS:
            reset_gyro_bias();
            break;
            
        case CMD_CONFIG:
            if (data != NULL && data_length > 0) {
                if (parse_config_data(data, data_length)) {
                    if (debug_mode) {
                        printf("Configuration updated successfully\n");
                    }
                    
                    // Reconfigure all detected IMUs
                    for (int i = 0; i < imu_count; i++) {
                        configure_imu(i);
                    }
                } else {
                    if (debug_mode) {
                        printf("Failed to parse configuration data\n");
                    }
                }
            } else {
                if (debug_mode) {
                    printf("No configuration data provided\n");
                }
            }
            break;
            
        case CMD_TEST:
            if (debug_mode) {
                printf("Sending test binary packet\n");
            }
            fflush(stdout);
            
            // Very simple test packet
            uint8_t test_packet[] = {
                0xAA, 0xBB,     // Start markers
                0x00,           // Packet count
                0x01,           // 1 IMU
                0x00, 0x00, 0x00, 0x00,  // Timestamp
                0x00,           // IMU index
                // Simple identity quaternion values
                0x00, 0x00, 0x80, 0x3F,  // w = 1.0
                0x00, 0x00, 0x00, 0x00,  // x = 0.0
                0x00, 0x00, 0x00, 0x00,  // y = 0.0
                0x00, 0x00, 0x00, 0x00,  // z = 0.0
                0x3F,           // Checksum
                0xCC, 0xDD      // End markers
            };
            
            for (int i = 0; i < sizeof(test_packet); i++) {
                putchar(test_packet[i]);
            }
            fflush(stdout);
            break;
            
        default:
            if (debug_mode) {
                printf("Unknown command code: 0x%02X\n", cmd_code);
            }
            break;
    }
}

// Handle incoming binary commands
void handle_binary_commands() {
    static enum {
        WAITING_FOR_START1,
        WAITING_FOR_START2,
        READING_CMD_CODE,
        READING_LENGTH,
        READING_DATA,
        READING_CHECKSUM,
        READING_END1,
        READING_END2
    } state = WAITING_FOR_START1;
    
    static uint8_t cmd_code = 0;
    static uint8_t data_length = 0;
    static uint8_t cmd_data[CMD_BUFFER_SIZE];
    static uint8_t data_index = 0;
    static uint8_t checksum = 0;
    static uint8_t received_checksum = 0;
    
    // Check for incoming data
    int c = getchar_timeout_us(0);
    if (c == PICO_ERROR_TIMEOUT) {
        return;
    }
    
    // Process byte based on current state
    switch (state) {
        case WAITING_FOR_START1:
            if (c == CMD_PACKET_START_1) {  // 0xFA
                state = WAITING_FOR_START2;
            }
            break;
            
        case WAITING_FOR_START2:
            if (c == CMD_PACKET_START_2) {  // 0xFB
                state = READING_CMD_CODE;
            } else {
                state = WAITING_FOR_START1;  // Invalid sequence, reset
            }
            break;
            
        case READING_CMD_CODE:
            // This is the command code
            cmd_code = c;
            state = READING_LENGTH;
            break;
            
        case READING_LENGTH:
            // This is the data length
            data_length = c;
            checksum = cmd_code ^ data_length; // Start calculating checksum
            
            if (data_length > 0) {
                // We have data to read
                state = READING_DATA;
                data_index = 0;
            } else {
                // No data, go straight to checksum
                state = READING_CHECKSUM;
            }
            break;
            
        case READING_DATA:
            // Reading data bytes
            if (data_index < CMD_BUFFER_SIZE && data_index < data_length) {
                cmd_data[data_index++] = c;
                checksum ^= c; // Update checksum
                
                if (data_index >= data_length) {
                    // Finished reading all data
                    state = READING_CHECKSUM;
                }
            } else {
                // Buffer overflow or too much data - abort
                state = WAITING_FOR_START1;
            }
            break;
            
        case READING_CHECKSUM:
            // This is the checksum
            received_checksum = c;
            state = READING_END1;
            break;
            
        case READING_END1:
            // This should be the first end marker
            if (c == CMD_PACKET_END_1) {  // 0xFC
                state = READING_END2;
            } else {
                state = WAITING_FOR_START1; // Invalid, reset
            }
            break;
            
        case READING_END2:
            // This should be the second end marker
            if (c == CMD_PACKET_END_2) {  // 0xFD
                // Valid packet end
                if (checksum == received_checksum) {
                    // Checksum valid, process the command
                    process_binary_command(cmd_code, data_length > 0 ? cmd_data : NULL, data_length);
                }
            }
            
            // Always reset to initial state after complete packet
            state = WAITING_FOR_START1;
            break;
    }
}

// Parse configuration data from binary command
bool parse_config_data(const uint8_t* data, uint8_t data_length) {
    // Need at least 32 bytes (8 parameters * 4 bytes each)
    if (data_length < 32) {
        if (debug_mode) {
            printf("Config data too short: %d bytes\n", data_length);
        }
        return false;
    }
    
    // Extract settings
    // Process noise (float)
    memcpy(&settings.process_noise, &data[0], 4);
    
    // Measurement noise (float)
    memcpy(&settings.measurement_noise, &data[4], 4);
    
    // Accel range (int32)
    int32_t accel_range;
    memcpy(&accel_range, &data[8], 4);
    settings.accel_range = accel_range;
    
    // Gyro range (int32)
    int32_t gyro_range;
    memcpy(&gyro_range, &data[12], 4);
    settings.gyro_range = gyro_range;
    
    // Accel rate (int32)
    int32_t accel_rate;
    memcpy(&accel_rate, &data[16], 4);
    settings.accel_rate = accel_rate;
    
    // Gyro rate (int32)
    int32_t gyro_rate;
    memcpy(&gyro_rate, &data[20], 4);
    settings.gyro_rate = gyro_rate;
    
    // Update interval (int32)
    int32_t interval_ms;
    memcpy(&interval_ms, &data[24], 4);
    data_interval_ms = interval_ms > 0 ? interval_ms : 50; // Default to 50ms
    
    // Auto bias update flag (int32 used as boolean)
    int32_t auto_bias;
    memcpy(&auto_bias, &data[28], 4);
    settings.auto_bias_update = auto_bias != 0;
    
    // Verbose flag (optional)
    if (data_length >= 36) {
        int32_t verbose;
        memcpy(&verbose, &data[32], 4);
        verbose_output = verbose != 0;
    }
    
    if (debug_mode) {
        printf("Updated settings:\n");
        printf("  Process noise: %.3f\n", settings.process_noise);
        printf("  Measurement noise: %.3f\n", settings.measurement_noise);
        printf("  Accel range: %d\n", settings.accel_range);
        printf("  Gyro range: %d\n", settings.gyro_range);
        printf("  Accel rate: %d\n", settings.accel_rate);
        printf("  Gyro rate: %d\n", settings.gyro_rate);
        printf("  Data interval: %llu ms\n", data_interval_ms);
        printf("  Auto bias updates: %s\n", settings.auto_bias_update ? "enabled" : "disabled");
        printf("  Verbose mode: %s\n", verbose_output ? "enabled" : "disabled");
    }
    
    return true;
}

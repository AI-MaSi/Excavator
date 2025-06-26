#include "imu_read.h"
#include "kalman_processing.h"
#include "imu_setup.h"

// Detect static vs. moving state (for gyro bias estimation)
void detect_motion_state(float dt) {
    for (int i = 0; i < imu_count; i++) {
        // Calculate gyro magnitude
        float gx = imus[i].gyro[0];
        float gy = imus[i].gyro[1];
        float gz = imus[i].gyro[2];
        float gyro_mag = sqrtf(gx*gx + gy*gy + gz*gz);

        // Check if static
        if (gyro_mag < motion_threshold) {
            if (!is_static[i]) {
                is_static[i] = true;
                static_duration[i] = 0;
            } else {
                static_duration[i] += dt;
            }
        } else {
            is_static[i] = false;
            static_duration[i] = 0;
        }
    }
}

// Check if an IMU is still responding
bool is_imu_alive(int imu_index) {
    if (!select_mux_channel(imus[imu_index].channel)) {
        return false;
    }
    
    uint8_t addr = imus[imu_index].address;
    uint8_t who_am_i_reg = ISM330_REG_WHO_AM_I;
    uint8_t data;
    
    int ret = i2c_write_blocking(I2C_PORT, addr, &who_am_i_reg, 1, true);
    if (ret < 0) return false;
    
    ret = i2c_read_blocking(I2C_PORT, addr, &data, 1, false);
    if (ret < 0) return false;
    
    return (data == ISM330_ID);
}

// Read data from all IMUs
bool read_imu_data() {
    static bool is_synthetic[MAX_IMUS] = {false};
    bool any_successful_read = false;
    static uint64_t last_debug_print = 0;
    uint64_t current_time = time_us_64();
    
    for (int i = 0; i < imu_count; i++) {
        if (!imus[i].active) continue;

        // If we determined this is a synthetic IMU, don't try hardware access
        if (!is_synthetic[i]) {
            bool read_success = true;
            
            // Select multiplexer channel
            if (!select_mux_channel(imus[i].channel)) {
                if (debug_mode && (current_time - last_debug_print > 3000000)) {
                    printf("Failed to select mux channel %d for IMU %d\n", imus[i].channel, i);
                }
                read_success = false;
            }

            if (read_success) {
                uint8_t addr = imus[i].address;
                uint8_t reg;
                uint8_t data[6];

                // Read gyroscope data
                reg = ISM330_OUTX_L_G;
                if (i2c_write_blocking(I2C_PORT, addr, &reg, 1, true) < 0) {
                    if (debug_mode && (current_time - last_debug_print > 3000000)) {
                        printf("Failed to write gyro reg addr to IMU %d\n", i);
                    }
                    read_success = false;
                }

                if (read_success && i2c_read_blocking(I2C_PORT, addr, data, 6, false) < 0) {
                    if (debug_mode && (current_time - last_debug_print > 3000000)) {
                        printf("Failed to read gyro data from IMU %d\n", i);
                    }
                    read_success = false;
                }

                if (read_success) {
                    // Convert gyro data (16-bit signed, LSB first)
                    int16_t gx = (int16_t)(data[1] << 8 | data[0]);
                    int16_t gy = (int16_t)(data[3] << 8 | data[2]);
                    int16_t gz = (int16_t)(data[5] << 8 | data[4]);

                    // Convert to degrees per second based on selected range
                    float gyro_scale;
                    switch (settings.gyro_range) {
                        case 0: gyro_scale = 8.75f / 1000.0f; break;  // 250 dps
                        case 1: gyro_scale = 17.5f / 1000.0f; break;  // 500 dps
                        case 2: gyro_scale = 35.0f / 1000.0f; break;  // 1000 dps
                        case 3: gyro_scale = 70.0f / 1000.0f; break;  // 2000 dps
                        case 125: gyro_scale = 4.375f / 1000.0f; break; // 125 dps
                        default: gyro_scale = 35.0f / 1000.0f; break;
                    }

                    // Apply bias correction directly from Kalman filter state
                    imus[i].gyro[0] = gx * gyro_scale - imus[i].x_data[4];
                    imus[i].gyro[1] = gy * gyro_scale - imus[i].x_data[5];
                    imus[i].gyro[2] = gz * gyro_scale - imus[i].x_data[6];

                    // Read accelerometer data
                    reg = ISM330_OUTX_L_XL;
                    if (i2c_write_blocking(I2C_PORT, addr, &reg, 1, true) < 0) {
                        if (debug_mode && (current_time - last_debug_print > 3000000)) {
                            printf("Failed to write accel reg addr to IMU %d\n", i);
                        }
                        read_success = false;
                    }

                    if (read_success && i2c_read_blocking(I2C_PORT, addr, data, 6, false) < 0) {
                        if (debug_mode && (current_time - last_debug_print > 3000000)) {
                            printf("Failed to read accel data from IMU %d\n", i);
                        }
                        read_success = false;
                    }

                    if (read_success) {
                        // Convert accel data (16-bit signed, LSB first)
                        int16_t ax = (int16_t)(data[1] << 8 | data[0]);
                        int16_t ay = (int16_t)(data[3] << 8 | data[2]);
                        int16_t az = (int16_t)(data[5] << 8 | data[4]);

                        // Convert to g based on selected range
                        float accel_scale;
                        switch (settings.accel_range) {
                            case 0: accel_scale = 0.061f / 1000.0f; break;  // 2g
                            case 2: accel_scale = 0.122f / 1000.0f; break;  // 4g
                            case 3: accel_scale = 0.244f / 1000.0f; break;  // 8g
                            case 1: accel_scale = 0.488f / 1000.0f; break;  // 16g
                            default: accel_scale = 0.061f / 1000.0f; break;
                        }

                        imus[i].accel[0] = ax * accel_scale;
                        imus[i].accel[1] = ay * accel_scale;
                        imus[i].accel[2] = az * accel_scale;
                        
                        any_successful_read = true;
                        
                        // Print raw values occasionally if in debug mode
                        if (debug_mode && (current_time - last_debug_print > 3000000)) {
                            printf("IMU %d hardware data: GYRO(x:%.4f, y:%.4f, z:%.4f) ACCEL(x:%.4f, y:%.4f, z:%.4f)\n", 
                                  i, imus[i].gyro[0], imus[i].gyro[1], imus[i].gyro[2], 
                                  imus[i].accel[0], imus[i].accel[1], imus[i].accel[2]);
                        }
                    }
                }
            }
            
            // If reading failed, mark as synthetic and use synthetic data
            if (!read_success) {
                is_synthetic[i] = true;
                if (debug_mode && (current_time - last_debug_print > 3000000)) {
                    printf("Switching IMU %d to synthetic data mode\n", i);
                }
            }
        }
        
        // Generate synthetic data if needed
        if (is_synthetic[i]) {
            // Zero gyro values to prevent quaternion spinning
            imus[i].gyro[0] = 0.0f;
            imus[i].gyro[1] = 0.0f;
            imus[i].gyro[2] = 0.0f;
            
            // Simulate gravity pointing down (1g in z direction)
            imus[i].accel[0] = 0.0f;
            imus[i].accel[1] = 0.0f;
            imus[i].accel[2] = 1.0f;
            
            if (debug_mode && (current_time - last_debug_print > 3000000)) {
                printf("IMU %d using synthetic data\n", i);
            }
        }
        
        // Periodically check if synthetic IMUs can be recovered (every ~5 seconds)
        if (is_synthetic[i] && (current_time % 5000000) < 1000) {
            if (is_imu_alive(i)) {
                if (debug_mode) {
                    printf("IMU %d recovered from synthetic mode\n", i);
                }
                is_synthetic[i] = false;
            }
        }
    }
    
    if (debug_mode && (current_time - last_debug_print > 3000000)) {
        last_debug_print = current_time;
    }
    
    sensor_read_count++;

    // Print performance stats occasionally
    if (debug_mode) {
        if (current_time - last_perf_print > 5000000) { // Every 5 seconds
            float elapsed = (current_time - last_perf_print) / 1000000.0f;
            float rate = sensor_read_count / elapsed;
            printf("IMU read rate: %.1f Hz, Using hardware: %s\n", 
                  rate, any_successful_read ? "YES" : "NO (synthetic data)");
            sensor_read_count = 0;
            last_perf_print = current_time;
        }
    }

    return true;
}

// Process all IMU data
void process_imu_data() {
    // Calculate time step
    uint64_t current_time = time_us_64();
    float dt = (current_time - last_read_time) / 1000000.0f;  // Convert to seconds

    // Constrain dt to reasonable values
    if (dt > 0.1f) dt = 0.01f;  // Default to 100Hz if long gap
    dt = fmaxf(fminf(dt, 0.05f), 0.001f);  // Constrain between 1ms and 50ms
    last_dt = dt;
    last_read_time = current_time;

    // Detect motion state (for bias updating)
    detect_motion_state(dt);

    // Update quaternions using Kalman filter
    update_quaternions_kalman();
}
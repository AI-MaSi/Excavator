#include "imu_setup.h"
#include "kalman_processing.h"

// Initialize I2C communication
bool init_i2c() {
    i2c_init(I2C_PORT, I2C_FREQ);
    gpio_set_function(I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_PIN);
    gpio_pull_up(I2C_SCL_PIN);

    // Add a small delay to stabilize
    sleep_ms(50);
    
    if (debug_mode) {
        printf("I2C initialized on SDA=%d, SCL=%d at %d Hz\n", I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ);
    }

    return true;
}

// Scan for a specific I2C device
bool scan_i2c_device(uint8_t addr) {
    uint8_t rxdata;
    int ret = i2c_read_blocking(I2C_PORT, addr, &rxdata, 1, false);
    
    if (debug_mode) {
        if (ret >= 0) {
            printf("Device found at I2C address 0x%02X\n", addr);
        }
    }
    
    return (ret >= 0);
}

// Find multiplexer - it could be at addresses 0x70-0x77
bool find_multiplexer() {
    if (debug_mode) {
        printf("Searching for multiplexer in address range 0x70-0x77...\n");
    }
    
    // First try the default address
    if (scan_i2c_device(TCA9548A_BASE_ADDR)) {
        mux_address = TCA9548A_BASE_ADDR;
        if (debug_mode) {
            printf("Found multiplexer at default address 0x%02X\n", mux_address);
        }
        return true;
    }
    
    // If not found at default, try the whole range
    for (uint8_t addr = 0x70; addr <= 0x77; addr++) {
        if (addr == TCA9548A_BASE_ADDR) continue; // Already checked
        
        if (scan_i2c_device(addr)) {
            mux_address = addr;
            if (debug_mode) {
                printf("Found multiplexer at address 0x%02X\n", mux_address);
            }
            return true;
        }
    }
    
    if (debug_mode) {
        printf("ERROR: Multiplexer not found in address range 0x70-0x77\n");
    }
    return false;
}

// Select TCA9548A multiplexer channel - keep delay short
bool select_mux_channel(uint8_t channel) {
    if (channel > 7) return false;
    
    uint8_t channel_mask = 1 << channel;
    int ret = i2c_write_blocking(I2C_PORT, mux_address, &channel_mask, 1, false);

    // Small delay to ensure channel switch completes
    sleep_us(10);  // Use the original short 10μs delay
    
    if (debug_mode && ret <= 0) {
        static uint64_t last_mux_error = 0;
        uint64_t current_time = time_us_64();
        
        if (current_time - last_mux_error > 3000000) { // Don't spam logs
            printf("Failed to select mux channel %d, mask 0x%02X (ret=%d)\n", 
                  channel, channel_mask, ret);
            last_mux_error = current_time;
        }
    }
    
    return (ret > 0);
}

// Check if an IMU is present on this channel with this address
bool check_imu(uint8_t channel, uint8_t addr) {
    // Select channel
    if (!select_mux_channel(channel)) {
        return false;
    }
    
    // Read WHO_AM_I register
    uint8_t who_am_i_reg = ISM330_REG_WHO_AM_I;
    uint8_t data;
    
    // Send register address
    int ret = i2c_write_blocking(I2C_PORT, addr, &who_am_i_reg, 1, true);
    if (ret < 0) return false;
    
    // Short delay
    sleep_us(50);
    
    // Read register value
    ret = i2c_read_blocking(I2C_PORT, addr, &data, 1, false);
    if (ret < 0) return false;
    
    if (debug_mode) {
        printf("Channel %d, Address 0x%02X: WHO_AM_I = 0x%02X (expected 0x%02X)\n", 
               channel, addr, data, ISM330_ID);
    }
    
    return (data == ISM330_ID);
}

// Scan for IMUs on all multiplexer channels
bool scan_for_imus() {
    imu_count = 0;
    
    // First find the multiplexer
    if (!find_multiplexer()) {
        return false;
    }
    
    if (debug_mode) {
        printf("Scanning for IMUs at addresses 0x6A and 0x6B on all multiplexer channels...\n");
    }

    // Try all 8 multiplexer channels
    for (int channel = 0; channel < 8; channel++) {
        // Try both IMU addresses
        if (check_imu(channel, ISM330_ADDR_PRIMARY)) {
            if (imu_count < MAX_IMUS) {
                imus[imu_count].active = true;
                imus[imu_count].channel = channel;
                imus[imu_count].address = ISM330_ADDR_PRIMARY;

                // Initialize quaternion to identity [1,0,0,0]
                imus[imu_count].quaternion[0] = 1.0f;
                imus[imu_count].quaternion[1] = 0.0f;
                imus[imu_count].quaternion[2] = 0.0f;
                imus[imu_count].quaternion[3] = 0.0f;

                if (debug_mode) {
                    printf("Found IMU_%d on channel %d, address 0x%02X\n",
                          imu_count, channel, ISM330_ADDR_PRIMARY);
                }
                
                imu_count++;
            }
        }
        
        if (check_imu(channel, ISM330_ADDR_SECONDARY)) {
            if (imu_count < MAX_IMUS) {
                imus[imu_count].active = true;
                imus[imu_count].channel = channel;
                imus[imu_count].address = ISM330_ADDR_SECONDARY;

                // Initialize quaternion to identity [1,0,0,0]
                imus[imu_count].quaternion[0] = 1.0f;
                imus[imu_count].quaternion[1] = 0.0f;
                imus[imu_count].quaternion[2] = 0.0f;
                imus[imu_count].quaternion[3] = 0.0f;

                if (debug_mode) {
                    printf("Found IMU_%d on channel %d, address 0x%02X\n",
                          imu_count, channel, ISM330_ADDR_SECONDARY);
                }
                
                imu_count++;
            }
        }
    }

    if (debug_mode) {
        printf("Found %d IMUs\n", imu_count);
    }
    
    // Create a synthetic IMU if none found
    if (imu_count == 0) {
        if (debug_mode) {
            printf("No IMUs found, creating synthetic data IMU\n");
        }
        
        imus[0].active = true;
        imus[0].channel = 0;
        imus[0].address = ISM330_ADDR_PRIMARY;
        
        // Initialize quaternion to identity
        imus[0].quaternion[0] = 1.0f;
        imus[0].quaternion[1] = 0.0f;
        imus[0].quaternion[2] = 0.0f;
        imus[0].quaternion[3] = 0.0f;
        
        imu_count = 1;
    }
    
    // Initialize Kalman filters for all IMUs
    for (int i = 0; i < imu_count; i++) {
        init_kalman_filter(i);
    }
    
    return true;
}

// Configure an IMU with specified settings
bool configure_imu(int imu_index) {
    if (imu_index >= imu_count || !imus[imu_index].active)
        return false;
        
    // Check if this is a synthetic IMU (created as a fallback)
    if (imu_count == 1 && imu_index == 0 && imus[0].channel == 0 && 
        !check_imu(imus[0].channel, imus[0].address)) {
        if (debug_mode) {
            printf("Skipping configuration for synthetic IMU %d\n", imu_index);
        }
        return true;
    }

    // Select the appropriate multiplexer channel
    if (!select_mux_channel(imus[imu_index].channel)) {
        if (debug_mode) {
            printf("Failed to select mux channel %d for configuration\n", imus[imu_index].channel);
        }
        return false;
    }

    uint8_t addr = imus[imu_index].address;

    // Build control registers based on settings
    uint8_t ctrl1_val = (settings.accel_rate << 4) | settings.accel_range;
    uint8_t ctrl2_val = (settings.gyro_rate << 4) | settings.gyro_range;

    // Configure accelerometer
    uint8_t buffer[2];
    buffer[0] = ISM330_CTRL1_XL;
    buffer[1] = ctrl1_val;
    int ret = i2c_write_blocking(I2C_PORT, addr, buffer, 2, false);
    if (ret < 0) {
        if (debug_mode) {
            printf("Failed to configure accelerometer for IMU %d (ret=%d)\n", imu_index, ret);
        }
        return false;
    }

    // Short delay
    sleep_us(100);

    // Configure gyroscope
    buffer[0] = ISM330_CTRL2_G;
    buffer[1] = ctrl2_val;
    ret = i2c_write_blocking(I2C_PORT, addr, buffer, 2, false);
    if (ret < 0) {
        if (debug_mode) {
            printf("Failed to configure gyroscope for IMU %d (ret=%d)\n", imu_index, ret);
        }
        return false;
    }

    if (debug_mode) {
        printf("Configured IMU_%d with accel=0x%02X, gyro=0x%02X\n",
               imu_index, ctrl1_val, ctrl2_val);
    }

    return true;
}
#include "config.h"
#include "imu_setup.h"
#include "imu_read.h"
#include "kalman_processing.h"
#include "data_transfer.h"

// Define global variables
volatile bool running = false;
FilterSettings settings;
IMUConfig imus[MAX_IMUS];
int imu_count = 0;
uint8_t cmd_buffer[CMD_BUFFER_SIZE];
uint64_t last_read_time = 0;
float last_dt = 0.10f;  // Default to 100ms (~10Hz)
bool is_static[MAX_IMUS];
float static_duration[MAX_IMUS];
uint64_t last_bias_update = 0;
float motion_threshold = 0.01f;  // deg/s
int sensor_read_count = 0;
uint64_t last_perf_print = 0;
uint8_t mux_address = TCA9548A_BASE_ADDR;
uint64_t last_data_time = 0;
uint64_t data_interval_ms = 100; // Default 10Hz output rate
bool verbose_output = false;     // Default to true for visualizer
uint32_t packet_count = 0;      // For packet sequencing
bool debug_mode = false;         // Enable debug logging

// Set default settings
void set_default_settings() {
    settings.process_noise = 0.01f;        // Process noise for Kalman filter
    settings.measurement_noise = 0.1f;     // Measurement noise for Kalman filter
    settings.auto_bias_update = false;     // Disable automatic bias updates
    settings.accel_range = 0;              // 2g
    settings.gyro_range = 125;             // 125 dps
    settings.accel_rate = 4;               // 104 Hz
    settings.gyro_rate = 4;                // 104 Hz
}

// Core 1 entry point - handles sensor reading and processing with precise timing
void core1_entry() {
    // Initialize sensor read timing
    last_read_time = time_us_64();
    
    // IMU read timing parameters
    uint64_t imu_read_interval_us = 0;     // Will be calculated based on configured rate
    uint64_t next_imu_read_time = 0;
    
    // Estimated time per IMU
    uint64_t time_per_imu_us = 500;        // Initial estimate: 500μs per IMU
    
    while (true) {
        uint64_t current_time = time_us_64();
        
        // Recalculate timing parameters when needed
        if (imu_read_interval_us == 0) {
            // Calculate interval based on sample rate setting
            int sample_rate_hz = 0;
            switch (settings.gyro_rate) {
                case 0: sample_rate_hz = 12; break;   // 12.5 Hz
                case 1: sample_rate_hz = 26; break;   // 26 Hz
                case 2: sample_rate_hz = 52; break;   // 52 Hz
                case 3: sample_rate_hz = 104; break;  // 104 Hz
                case 4: sample_rate_hz = 208; break;  // 208 Hz
                case 5: sample_rate_hz = 416; break;  // 416 Hz
                case 6: sample_rate_hz = 833; break;  // 833 Hz
                case 7: sample_rate_hz = 1666; break; // 1.66 kHz
                default: sample_rate_hz = 104; break; // Default to 104 Hz
            }
            
            // Calculate read interval in microseconds
            imu_read_interval_us = 1000000 / sample_rate_hz;
            
            // Set initial times if not set
            if (next_imu_read_time == 0) {
                next_imu_read_time = current_time;
            }
        }
        
        // Only process if running
        if (running) {
            // Time to read IMUs?
            if (current_time >= next_imu_read_time) {
                uint64_t read_start_time = time_us_64();
                
                // Read sensor data
                read_imu_data();

                // Process IMU data
                process_imu_data();
                
                // Send quaternion data if it's time
                send_quaternion_data();
                
                // Measure actual time taken
                uint64_t read_time = time_us_64() - read_start_time;
                
                // Update estimate of time per IMU (with smoothing)
                if (imu_count > 0) {
                    uint64_t new_time_per_imu = read_time / imu_count;
                    time_per_imu_us = (time_per_imu_us * 9 + new_time_per_imu) / 10; // Smoothed average
                }
                
                // Schedule next update
                next_imu_read_time += imu_read_interval_us;
                
                // If we've fallen too far behind, reset
                if (current_time > next_imu_read_time + imu_read_interval_us) {
                    next_imu_read_time = current_time + imu_read_interval_us;
                }
            }
        }
        
        // Calculate sleep time to avoid busy waiting
        uint64_t sleep_time = 0;
        
        if (next_imu_read_time > current_time) {
            sleep_time = next_imu_read_time - current_time;
            // Cap sleep time to avoid oversleeping
            if (sleep_time > 1000) {
                sleep_time = 1000; // Max 1ms sleep
            }
            sleep_us(sleep_time);
        } else {
            // Just yield a little CPU time
            sleep_us(10);
        }
    }
}

int main() {
    // Initialize stdio
    stdio_init_all();
    sleep_ms(1000);  // Wait for USB to initialize

    printf("\n\nRP2040 IMU Processor with Standard Kalman Filter\n");
    printf("Version: 1.1 - Modified with Synthetic Data Support\n");

    // Initialize I2C - must happen before anything else
    if (!init_i2c()) {
        printf("ERROR: Failed to initialize I2C\n");
        return -1;
    }

    // Set default settings
    set_default_settings();

    // Start with the multiplexer scan
    printf("Checking for multiplexer and IMUs...\n");
    
    // Similar to the working code, we'll scan for IMUs
    // The modified scan_for_imus will create a synthetic data IMU if no real IMUs are found
    if (scan_for_imus()) {
        printf("IMU scan completed, found %d IMUs\n", imu_count);
        
        // Configure all detected IMUs
        for (int i = 0; i < imu_count; i++) {
            configure_imu(i);
            printf("Configured IMU %d on channel %d, address 0x%02X\n", 
                  i, imus[i].channel, imus[i].address);
        }
    } else {
        printf("ERROR: Failed to scan for IMUs\n");
        return -1;
    }

    // Print detailed I2C debug information
    printf("I2C Configuration: Port=%d, SDA=%d, SCL=%d, Freq=%d Hz\n", 
           I2C_PORT == i2c1 ? 1 : 0, I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ);
    
    // Initialize motion detection arrays
    for (int i = 0; i < MAX_IMUS; i++) {
        is_static[i] = false;
        static_duration[i] = 0.0f;
    }

    // Start core 1 for sensor reading
    multicore_launch_core1(core1_entry);
    
    printf("Binary command protocol ready\n");
    printf("Format: FA FB [cmd] [len] [data...] [csum] FC FD\n");

    // Main loop on core 0 - handle commands
    while (true) {
        // Handle binary commands
        handle_binary_commands();

        // Small delay to avoid spinning too fast
        sleep_ms(1);
    }

    return 0;
}
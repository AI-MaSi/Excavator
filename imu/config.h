// IMU Processor Configuration
// Shared constants, typedefs, and global declarations

#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"

// Include Kalman filter headers
//#include <kalman.h>
//#include <matrix.h>
//#include <cholesky.h>

// use the wrapper instead
#include "kalman_wrapper.h"

// Constants
#define MAX_IMUS 8
#define CMD_BUFFER_SIZE 128

// Binary command codes
#define CMD_START      0x01
#define CMD_STOP       0x02
#define CMD_SCAN       0x03
#define CMD_STATUS     0x04
#define CMD_RESETBIAS  0x05
#define CMD_CONFIG     0x06
#define CMD_TEST       0x07

// Sample rate definitions in Hz
#define SAMPLE_RATE_0   12.5f   // 12.5 Hz
#define SAMPLE_RATE_1   26.0f   // 26 Hz
#define SAMPLE_RATE_2   52.0f   // 52 Hz
#define SAMPLE_RATE_3   104.0f  // 104 Hz
#define SAMPLE_RATE_4   208.0f  // 208 Hz
#define SAMPLE_RATE_5   416.0f  // 416 Hz
#define SAMPLE_RATE_6   833.0f  // 833 Hz
#define SAMPLE_RATE_7   1666.0f // 1.66 kHz

// I2C definitions
#define I2C_PORT i2c1
#define I2C_SDA_PIN 6   // D4
#define I2C_SCL_PIN 7   // D5
#define I2C_FREQ 400000  // 400 kHz

// IMU Registers
#define ISM330_ADDR_PRIMARY 0x6A
#define ISM330_ADDR_SECONDARY 0x6B
#define ISM330_REG_WHO_AM_I 0x0F
#define ISM330_ID 0x6B
#define ISM330_CTRL1_XL 0x10  // Accelerometer control
#define ISM330_CTRL2_G 0x11   // Gyroscope control
#define ISM330_OUTX_L_G 0x22  // First gyro register
#define ISM330_OUTX_L_XL 0x28 // First accel register

// TCA9548A multiplexer
#define TCA9548A_BASE_ADDR 0x70

// Data packet markers for data packets (output)
#define DATA_PACKET_START_1 0xAA
#define DATA_PACKET_START_2 0xBB
#define DATA_PACKET_END_1 0xCC
#define DATA_PACKET_END_2 0xDD

// Binary protocol markers for command packets (input)
#define CMD_PACKET_START_1 0xFA
#define CMD_PACKET_START_2 0xFB
#define CMD_PACKET_END_1 0xFC
#define CMD_PACKET_END_2 0xFD

// IMU state dimensions
#define NUM_STATES 7       // 4 quaternion components + 3 gyro bias
#define NUM_INPUTS 3       // 3 gyro measurements
#define NUM_MEASUREMENTS 3 // 3 accelerometer measurements

// Define matrix data type
typedef float matrix_data_t;

// Settings structure
typedef struct {
    float process_noise;      // Process noise (Q)
    float measurement_noise;  // Measurement noise (R)
    bool auto_bias_update;    // Flag to control automatic bias updating
    int accel_range;          // Accelerometer range setting
    int gyro_range;           // Gyroscope range setting
    int accel_rate;           // Accelerometer sample rate
    int gyro_rate;            // Gyroscope sample rate
} FilterSettings;

// IMU configuration with standard Kalman filter
typedef struct {
    bool active;
    uint8_t channel;
    uint8_t address;
    float accel[3];             // Raw accelerometer data
    float gyro[3];              // Raw gyroscope data
    float quaternion[4];        // Orientation quaternion (w, x, y, z)
    
    // Kalman filter structures
    kalman_t kf;                // Main Kalman filter
    kalman_measurement_t kfm;   // Measurement update structure
    
    // State vector and matrices
    matrix_data_t x_data[NUM_STATES];             // [qw, qx, qy, qz, bias_x, bias_y, bias_z]
    matrix_data_t A_data[NUM_STATES*NUM_STATES];  // State transition matrix
    matrix_data_t B_data[NUM_STATES*NUM_INPUTS];  // Input matrix (gyro -> state)
    matrix_data_t u_data[NUM_INPUTS];             // Input vector (gyro measurements)
    matrix_data_t P_data[NUM_STATES*NUM_STATES];  // Error covariance matrix
    matrix_data_t Q_data[NUM_INPUTS*NUM_INPUTS];  // Process noise covariance
    
    // Measurement matrices
    matrix_data_t H_data[NUM_MEASUREMENTS*NUM_STATES];  // Measurement matrix (accelerometer)
    matrix_data_t z_data[NUM_MEASUREMENTS];             // Measurement vector (normalized gravity)
    matrix_data_t R_data[NUM_MEASUREMENTS*NUM_MEASUREMENTS];  // Measurement noise covariance
    matrix_data_t y_data[NUM_MEASUREMENTS];             // Innovation vector
    matrix_data_t S_data[NUM_MEASUREMENTS*NUM_MEASUREMENTS];  // Innovation covariance
    matrix_data_t K_data[NUM_STATES*NUM_MEASUREMENTS];  // Kalman gain
    
    // Auxiliary buffers for matrix operations
    matrix_data_t aux_data[NUM_STATES > NUM_MEASUREMENTS ? NUM_STATES : NUM_MEASUREMENTS];  // Auxiliary buffer
    matrix_data_t predicted_x_data[NUM_STATES];         // Predicted state
    matrix_data_t temp_P_data[NUM_STATES*NUM_STATES];   // Temporary P matrix
    matrix_data_t temp_BQ_data[NUM_STATES*NUM_INPUTS];  // Temporary BQ matrix
    matrix_data_t S_inv_data[NUM_MEASUREMENTS*NUM_MEASUREMENTS];  // Inverted S matrix
    matrix_data_t temp_HP_data[NUM_MEASUREMENTS*NUM_STATES];      // Temporary HP matrix
    matrix_data_t temp_PHt_data[NUM_STATES*NUM_MEASUREMENTS];     // Temporary PH' matrix
    matrix_data_t temp_KHP_data[NUM_STATES*NUM_STATES];  // Temporary KHP matrix
} IMUConfig;

// Global variables (declared here, defined in main.c)
extern volatile bool running;
extern FilterSettings settings;
extern IMUConfig imus[MAX_IMUS];
extern int imu_count;
extern uint8_t cmd_buffer[CMD_BUFFER_SIZE];
extern uint64_t last_read_time;
extern float last_dt;
extern bool is_static[MAX_IMUS];
extern float static_duration[MAX_IMUS];
extern uint64_t last_bias_update;
extern float motion_threshold;
extern int sensor_read_count;
extern uint64_t last_perf_print;
extern uint8_t mux_address;
extern uint64_t last_data_time;
extern uint64_t data_interval_ms;
extern bool verbose_output;
extern uint32_t packet_count;
extern bool debug_mode;

#endif // CONFIG_H

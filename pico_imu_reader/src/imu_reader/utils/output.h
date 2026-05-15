#ifndef OUTPUT_H
#define OUTPUT_H

#include <stdbool.h>
#include <inttypes.h>
#include "imu_reader.h"

#define MAX_SENSORS 4
#define FLOATS_PER_SENSOR 7

// Message types for control frames (version=2)
#define MSG_TYPE_ERR_I2C   0x05  // I2C bus init failed
#define MSG_TYPE_ERR_IMU   0x06  // No IMU detected
#define MSG_TYPE_CAL_WAIT  0x07  // Startup gyro-bias calibration in progress
#define MSG_TYPE_ERR_CAL   0x08  // Startup calibration failed (robot moved during cal)

#define CAL_REPORT_ACCEPTED 0x01

#define CAL_FAIL_INSUFFICIENT_SAMPLES 0x0001
#define CAL_FAIL_GYRO_STD            0x0002
#define CAL_FAIL_ACCEL_STD           0x0004
#define CAL_FAIL_ACCEL_NORM          0x0008

typedef struct calibration_vector_t {
    float x;
    float y;
    float z;
} calibration_vector_t;

typedef struct calibration_report_sensor_t {
    uint32_t sample_count;
    uint32_t read_error_count;
    uint16_t failure_flags;
    calibration_vector_t gyro_mean;
    calibration_vector_t gyro_std;
    calibration_vector_t accel_mean;
    calibration_vector_t accel_std;
    float accel_norm_mean;
    float accel_norm_min;
    float accel_norm_max;
} calibration_report_sensor_t;

void print_output_data(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR], uint8_t sensor_count);
void send_control_msg(uint8_t msg_type);
void send_descriptor_frame(uint16_t sample_rate_hz, uint8_t sensor_count, const uint8_t *bus_ids, const uint8_t *device_addrs);
void send_calibration_report(uint32_t duration_ms, uint8_t sensor_count, uint8_t flags,
                             float gyro_std_limit_dps, float accel_std_limit_g,
                             float accel_norm_min_g, float accel_norm_max_g,
                             const calibration_report_sensor_t *sensor_reports);

extern imu_reader_settings_t imu_reader_settings;

#endif

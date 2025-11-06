#ifndef IMU_READER_H
#define IMU_READER_H

#include "inttypes.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
//#include "tusb.h"
#include "time.h"

typedef struct imu_reader_settings_t {
    int sensorCount;
    int lpfEnabled;
    float lpf_alpha;
    int sampleRate;
    int channelCount;
    int pitchOnly; // 1 = output pitch-only quaternion, 0 = full AHRS quaternion
    int streamRaw; // 1 = include raw accel in stream
    int formatCSV; // 1 = compact CSV line (no labels)
} imu_reader_settings_t;

enum settings_enum_e {
    S_SENSOR_COUNT,
    S_LPF_ENABLED,
    S_LPF_ALPHA,
    S_SAMPLE_RATE,
    S_QMODE,
    S_STREAM_RAW,
    S_FORMAT
};
typedef enum settings_enum_e settings_enum;

// Global imu reader settings for the application
extern imu_reader_settings_t imu_reader_settings;
extern settings_enum settings_option;

#endif


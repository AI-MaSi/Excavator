#ifndef IMU_READER_H
#define IMU_READER_H

#include "inttypes.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

typedef struct imu_reader_settings_t {
    int sampleRate;
    float gyroRangeDps;
    float ahrsGain;
    float ahrsAccelRejection;
    float ahrsRecoveryPeriodS;
    float offsetTimeoutS;
} imu_reader_settings_t;

enum settings_enum_e {
    S_SAMPLE_RATE,
    S_GYRO_RANGE,
    S_AHRS_GAIN,
    S_AHRS_ACCEL_REJ,
    S_AHRS_RECOVERY_S,
    S_OFFSET_TIMEOUT_S
};
typedef enum settings_enum_e settings_enum;

// Global settings
extern imu_reader_settings_t imu_reader_settings;
extern settings_enum settings_option;

#endif

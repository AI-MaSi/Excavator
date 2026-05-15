#ifndef IMU_READER_H
#define IMU_READER_H

typedef struct imu_reader_settings_t {
    int   sampleRate;
    float gyroRangeDps;
    float ahrsGain;
    float ahrsAccelRejection;
    float ahrsRecoveryPeriodS;
} imu_reader_settings_t;

// Global settings (defined in settings.c, consumed by InitFusion.c)
extern imu_reader_settings_t imu_reader_settings;

#endif

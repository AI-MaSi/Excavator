#include "InitFusion.h"
#include "pico/time.h"

extern imu_reader_settings_t imu_reader_settings;

void initialize_sensors_values(Sensor* sensors, int count) {
    for (int i = 0; i < count; i++) {
        sensors[i].gyroscope.axis.x = 0.0f;
        sensors[i].gyroscope.axis.y = 0.0f;
        sensors[i].gyroscope.axis.z = 0.0f;

        sensors[i].accelerometer.axis.x = 0.0f;
        sensors[i].accelerometer.axis.y = 0.0f;
        sensors[i].accelerometer.axis.z = 0.0f;

        sensors[i].mountingQuaternion = FUSION_IDENTITY_QUATERNION;
        sensors[i].lastPitchDeg = 0.0f;
        sensors[i].unwrapOffsetDeg = 0.0f;
        sensors[i].zeroOffsetDeg = 0.0f;
        sensors[i].pitchDeg = 0.0f;

        // Initialize timestamps to avoid huge deltaTime on first iteration
        sensors[i].timestamp = time_us_64();
        sensors[i].previousTimestamp = sensors[i].timestamp;

        // Initialize previous quaternion for continuity tracking
        sensors[i].previousQuaternion = FUSION_IDENTITY_QUATERNION;
    }
}

void initialize_calibrations(Sensor* sensors, int count) {
    for (int i = 0; i < count; i++) {
        // Identity calibration (no correction)
        sensors[i].calibration.gyroscopeMisalignment = (FusionMatrix){
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        };
        sensors[i].calibration.gyroscopeSensitivity = (FusionVector){1.0f, 1.0f, 1.0f};
        sensors[i].calibration.gyroscopeOffset = (FusionVector){0.0f, 0.0f, 0.0f};

        sensors[i].calibration.accelerometerMisalignment = (FusionMatrix){
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        };
        sensors[i].calibration.accelerometerSensitivity = (FusionVector){1.0f, 1.0f, 1.0f};
        sensors[i].calibration.accelerometerOffset = (FusionVector){0.0f, 0.0f, 0.0f};
    }
}

void initialize_algos(Sensor* sensors, int count) {
    for (int i = 0; i < count; i++) {
        // Initialize gyro offset tracker
        FusionOffsetInitialise(&sensors[i].offset, imu_reader_settings.sampleRate);
        if (imu_reader_settings.offsetTimeoutS > 0.0f) {
            sensors[i].offset.timeout = (unsigned int)(imu_reader_settings.offsetTimeoutS * imu_reader_settings.sampleRate);
            if (sensors[i].offset.timeout < 1) {
                sensors[i].offset.timeout = 1;
            }
        }

        // Initialize AHRS
        FusionAhrsInitialise(&sensors[i].ahrs);

        // Recovery period in samples
        float recovery_s = imu_reader_settings.ahrsRecoveryPeriodS;
        if (recovery_s <= 0.0f) recovery_s = 1.0f;

        sensors[i].settings = (FusionAhrsSettings){
            .convention = FusionConventionNwu,
            .gain = imu_reader_settings.ahrsGain,
            .gyroscopeRange = imu_reader_settings.gyroRangeDps,
            .accelerationRejection = imu_reader_settings.ahrsAccelRejection,
            .magneticRejection = 0.0f,  // No magnetometer
            .recoveryTriggerPeriod = (unsigned int)(recovery_s * (float)imu_reader_settings.sampleRate),
        };
        FusionAhrsSetSettings(&sensors[i].ahrs, &sensors[i].settings);
    }
}

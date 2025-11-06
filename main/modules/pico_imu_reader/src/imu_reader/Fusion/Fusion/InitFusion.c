#include "InitFusion.h"

extern imu_reader_settings_t imu_reader_settings; 

void initialize_sensors_values(Sensor* sensors) {
    for (int i = 0; i<imu_reader_settings.sensorCount;i++) {
        sensors[i].gyroscope.axis.x = 0.0f;
        sensors[i].gyroscope.axis.y = 0.0f;
        sensors[i].gyroscope.axis.z = 0.0f;
        sensors[i].gyroscope_old.axis.x = 0.0f;
        sensors[i].gyroscope_old.axis.y = 0.0f;
        sensors[i].gyroscope_old.axis.z = 0.0f;

        sensors[i].accelerometer.axis.x = 0.0f;
        sensors[i].accelerometer.axis.y = 0.0f;
        sensors[i].accelerometer.axis.z = 0.0f;
        sensors[i].accelerometer_old.axis.x = 0.0f;
        sensors[i].accelerometer_old.axis.y = 0.0f;
        sensors[i].accelerometer_old.axis.z = 0.0f;

        sensors[i].mountingQuaternion = FUSION_IDENTITY_QUATERNION;
        sensors[i].lastPitchDeg = 0.0f;
        sensors[i].unwrapOffsetDeg = 0.0f;
        sensors[i].zeroOffsetDeg = 0.0f;
        sensors[i].pitchDeg = 0.0f;
    }
}

void initialize_calibrations(Sensor* sensors) {
    for (int i = 0; i<imu_reader_settings.sensorCount;i++) {
        sensors[i].calibration.gyroscopeMisalignment = (FusionMatrix){ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        sensors[i].calibration.gyroscopeSensitivity = (FusionVector){1.0f, 1.0f, 1.0f};
        sensors[i].calibration.gyroscopeOffset = (FusionVector) {0.0f, 0.0f, 0.0f};
        sensors[i].calibration.accelerometerMisalignment = (FusionMatrix) {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        sensors[i].calibration.accelerometerSensitivity = (FusionVector) {1.0f, 1.0f, 1.0f};
        sensors[i].calibration.accelerometerOffset = (FusionVector) {0.0f, 0.0f, 0.0f};
    }
}

void initialize_algos(Sensor* sensors) {
    for (int i = 0; i<imu_reader_settings.sensorCount;i++) {
        FusionOffsetInitialise(&sensors[i].offset, imu_reader_settings.sampleRate);
        FusionAhrsInitialise(&sensors[i].ahrs);
        // Set AHRS algorithm settings
        /*
         * Tuning for excavator IK with possible slew interference:
         * - Lower gain leans more on gyro during transients (snappier, less accel-induced tilt).
         * - Higher accelerationRejection tells AHRS to ignore lateral/centripetal accel more often.
         * - Shorter recoveryTriggerPeriod resumes accel use sooner after transients (keeps drift bounded).
         * If pitch looks sluggish, increase gain (e.g., 0.4–0.5) and/or reduce accelerationRejection (8–12).
         */
        sensors[i].settings = (FusionAhrsSettings){
                .convention = FusionConventionNwu,
                .gain = 0.35f,
                .gyroscopeRange = 250.0f, /* match IMU dps range */
                .accelerationRejection = 14.0f,
                .magneticRejection = 0.0f, /* mag unused */
                .recoveryTriggerPeriod = 3 * imu_reader_settings.sampleRate, /* ~3 s */
        };
        FusionAhrsSetSettings(&sensors[i].ahrs, &sensors[i].settings);
    }
}

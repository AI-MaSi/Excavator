#ifndef FUSIONSTRUCTS_H
#define FUSIONSTRUCTS_H

#include "Fusion.h"
#include <time.h>

typedef struct FusionCalibration {
    FusionMatrix gyroscopeMisalignment;
    FusionVector gyroscopeSensitivity;
    FusionVector gyroscopeOffset;
    FusionMatrix accelerometerMisalignment;
    FusionVector accelerometerSensitivity;
    FusionVector accelerometerOffset;
} FusionCalibration;

typedef struct Sensor {
    FusionCalibration calibration;
    FusionAhrs ahrs;
    FusionOffset offset;
    FusionAhrsSettings settings;
    FusionVector accelerometer;
    FusionVector accelerometer_old;
    FusionVector gyroscope;
    FusionVector gyroscope_old;
    uint64_t previousTimestamp;
    uint64_t timestamp;
    FusionVector magnetometer;
    // Mounting transform from sensor frame to boom local frame
    FusionQuaternion mountingQuaternion;
    // Pitch estimation helpers (degrees)
    float lastPitchDeg;
    float unwrapOffsetDeg;
    float zeroOffsetDeg;
    float pitchDeg;
} Sensor;

#endif // FUSIONSTRUCTS_H

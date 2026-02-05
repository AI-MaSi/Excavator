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
    FusionVector gyroscope;
    uint64_t previousTimestamp;
    uint64_t timestamp;
    // Mounting transform (unused but kept for future)
    FusionQuaternion mountingQuaternion;
    // Pitch tracking (degrees)
    float lastPitchDeg;
    float unwrapOffsetDeg;
    float zeroOffsetDeg;
    float pitchDeg;
    // Previous quaternion for hemisphere continuity
    FusionQuaternion previousQuaternion;
} Sensor;

#endif // FUSIONSTRUCTS_H

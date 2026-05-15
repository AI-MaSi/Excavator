#include "settings.h"

// AHRS settings are hardcoded — no host handshake needed.
// Pico self-calibrates on power-on (~30 s stationary), then streams at 200 Hz.
imu_reader_settings_t imu_reader_settings = {
    .sampleRate          = 200,
    .gyroRangeDps        = 250.0f,
    .ahrsGain            = 4.5f,
    .ahrsAccelRejection  = 20.0f,
    .ahrsRecoveryPeriodS = 0.5f,
};

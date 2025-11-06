#ifndef ISM330DLC_CONFIG_H
#define ISM330DLC_CONFIG_H

// Accelerometer
#define XL_ODR 0x50 // 208 Hz
#define XL_G_RANGE_MASK 0x00 // 2g
#define XL_G_RANGE 2

// Gyroscope
#define G_DPS_RANGE_MASK 0x00 // 250dps
#define G_DPS_RANGE 250
#define G_ODR 0x50 // 208 kHz
#endif //ISM330DLC_CONFIG_H

#ifndef IMU_SETUP_H
#define IMU_SETUP_H

#include "config.h"

// Function prototypes
bool init_i2c(void);
bool scan_i2c_device(uint8_t addr);
bool select_mux_channel(uint8_t channel);
bool scan_for_imus(void);
bool configure_imu(int imu_index);

#endif // IMU_SETUP_H

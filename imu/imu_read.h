#ifndef IMU_READ_H
#define IMU_READ_H

#include "config.h"

// Function prototypes
bool read_imu_data(void);
void detect_motion_state(float dt);
void process_imu_data(void);
bool is_imu_alive(int imu_index);

#endif // IMU_READ_H

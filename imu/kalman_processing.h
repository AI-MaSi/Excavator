#ifndef KALMAN_PROCESSING_H
#define KALMAN_PROCESSING_H

#include "config.h"

// Function prototypes
void init_kalman_filter(int imu_index);
void update_state_transition(int imu_index, float dt);
void update_H_matrix(int imu_index);
void normalize_quaternion(int imu_index);
void update_gyro_bias(int imu_index);
void extract_quaternion_from_state(int imu_index);
void update_quaternions_kalman(void);
void reset_gyro_bias(void);

#endif // KALMAN_PROCESSING_H

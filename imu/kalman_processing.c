#include "kalman_processing.h"

// Initialize Kalman filter
void init_kalman_filter(int imu_index) {
    IMUConfig* imu = &imus[imu_index];
    
    // Initialize state vector [qw, qx, qy, qz, bias_x, bias_y, bias_z]
    imu->x_data[0] = 1.0f; // Initial quaternion = identity
    imu->x_data[1] = 0.0f;
    imu->x_data[2] = 0.0f;
    imu->x_data[3] = 0.0f;
    imu->x_data[4] = 0.0f; // Initial gyro bias estimates = 0
    imu->x_data[5] = 0.0f;
    imu->x_data[6] = 0.0f;
    
    // Initialize quaternion
    imu->quaternion[0] = 1.0f;
    imu->quaternion[1] = 0.0f;
    imu->quaternion[2] = 0.0f;
    imu->quaternion[3] = 0.0f;
    
    // Setup state transition matrix (identity matrix to start)
    memset(imu->A_data, 0, NUM_STATES*NUM_STATES*sizeof(matrix_data_t));
    for (int i = 0; i < NUM_STATES; i++) {
        imu->A_data[i*NUM_STATES+i] = 1.0f;
    }
    
    // Setup input matrix (how gyro measurements affect state)
    memset(imu->B_data, 0, NUM_STATES*NUM_INPUTS*sizeof(matrix_data_t));
    
    // Map gyro inputs to quaternion rates (simplified approximation)
    // These will be properly updated in the update_state_transition function
    imu->B_data[1*NUM_STATES+0] = 0.5f; // x gyro -> qx rate
    imu->B_data[2*NUM_STATES+1] = 0.5f; // y gyro -> qy rate
    imu->B_data[3*NUM_STATES+2] = 0.5f; // z gyro -> qz rate
    
    // Setup process noise (Q) - how much we trust our model
    memset(imu->Q_data, 0, NUM_INPUTS*NUM_INPUTS*sizeof(matrix_data_t));
    float gyro_noise = settings.process_noise;
    imu->Q_data[0*NUM_INPUTS+0] = gyro_noise;
    imu->Q_data[1*NUM_INPUTS+1] = gyro_noise;
    imu->Q_data[2*NUM_INPUTS+2] = gyro_noise;
    
    // Setup error covariance matrix (P)
    memset(imu->P_data, 0, NUM_STATES*NUM_STATES*sizeof(matrix_data_t));
    for (int i = 0; i < 4; i++) {
        imu->P_data[i*NUM_STATES+i] = 1.0f; // Initial quaternion uncertainty
    }
    for (int i = 4; i < NUM_STATES; i++) {
        imu->P_data[i*NUM_STATES+i] = 0.01f; // Initial bias uncertainty
    }
    
    // Setup measurement matrix (H) - maps state to accelerometer
    memset(imu->H_data, 0, NUM_MEASUREMENTS*NUM_STATES*sizeof(matrix_data_t));
    // Will be updated dynamically based on current quaternion state
    
    // Setup measurement noise (R) - how much we trust accelerometer
    memset(imu->R_data, 0, NUM_MEASUREMENTS*NUM_MEASUREMENTS*sizeof(matrix_data_t));
    float accel_noise = settings.measurement_noise;
    imu->R_data[0*NUM_MEASUREMENTS+0] = accel_noise;
    imu->R_data[1*NUM_MEASUREMENTS+1] = accel_noise;
    imu->R_data[2*NUM_MEASUREMENTS+2] = accel_noise;
    
    // Initialize other vectors
    memset(imu->z_data, 0, NUM_MEASUREMENTS*sizeof(matrix_data_t));
    memset(imu->u_data, 0, NUM_INPUTS*sizeof(matrix_data_t));
    
    // Initialize the Kalman filter structures using the standard Kalman library
    kalman_filter_initialize(
        &imu->kf,                     // Kalman filter structure
        NUM_STATES,                   // Number of states
        NUM_INPUTS,                   // Number of inputs
        imu->A_data,                  // State transition matrix
        imu->x_data,                  // State vector
        imu->B_data,                  // Input matrix
        imu->u_data,                  // Input vector
        imu->P_data,                  // State covariance matrix
        imu->Q_data,                  // Process noise matrix
        imu->aux_data,                // Auxiliary buffer
        imu->predicted_x_data,        // Predicted state
        imu->temp_P_data,             // Temporary P matrix
        imu->temp_BQ_data             // Temporary BQ matrix
    );
    
    kalman_measurement_initialize(
        &imu->kfm,                    // Kalman measurement structure
        NUM_STATES,                   // Number of states
        NUM_MEASUREMENTS,             // Number of measurements
        imu->H_data,                  // Measurement matrix
        imu->z_data,                  // Measurement vector
        imu->R_data,                  // Measurement noise matrix
        imu->y_data,                  // Innovation vector
        imu->S_data,                  // Innovation covariance
        imu->K_data,                  // Kalman gain
        imu->aux_data,                // Auxiliary buffer
        imu->S_inv_data,              // Inverted S matrix
        imu->temp_HP_data,            // Temporary HP matrix
        imu->temp_PHt_data,           // Temporary PH' matrix
        imu->temp_KHP_data            // Temporary KHP matrix
    );
    
    if (debug_mode) {
        printf("Initialized Kalman filter for IMU %d\n", imu_index);
    }
}

// Update the state transition matrix based on current quaternion and dt
void update_state_transition(int imu_index, float dt) {
    IMUConfig* imu = &imus[imu_index];
    
    // Extract current quaternion
    float q0 = imu->x_data[0]; // w
    float q1 = imu->x_data[1]; // x
    float q2 = imu->x_data[2]; // y
    float q3 = imu->x_data[3]; // z
    
    // Clear A matrix and set identity
    memset(imu->A_data, 0, NUM_STATES*NUM_STATES*sizeof(matrix_data_t));
    for (int i = 0; i < NUM_STATES; i++) {
        imu->A_data[i*NUM_STATES+i] = 1.0f;
    }
    
    // Update quaternion rotation part based on gyro input - linearized approximation
    // This maps how gyro rates affect quaternion components
    float half_dt = 0.5f * dt;
    
    // dq/dt = 0.5 * q ⊗ ω
    // Row 0 (qw) <- affected by [qx,qy,qz] * [gx,gy,gz]
    imu->A_data[0*NUM_STATES+1] += -half_dt;  // -qx/dt
    imu->A_data[0*NUM_STATES+2] += -half_dt;  // -qy/dt
    imu->A_data[0*NUM_STATES+3] += -half_dt;  // -qz/dt
    
    // Row 1 (qx) <- affected by [qw,qy,qz] * [gx,gy,gz]
    imu->A_data[1*NUM_STATES+0] += half_dt;   // qw/dt
    imu->A_data[1*NUM_STATES+2] += half_dt;   // qz/dt
    imu->A_data[1*NUM_STATES+3] += -half_dt;  // -qy/dt
    
    // Row 2 (qy) <- affected by [qw,qx,qz] * [gx,gy,gz]
    imu->A_data[2*NUM_STATES+0] += half_dt;   // qw/dt
    imu->A_data[2*NUM_STATES+1] += -half_dt;  // -qz/dt
    imu->A_data[2*NUM_STATES+3] += half_dt;   // qx/dt
    
    // Row 3 (qz) <- affected by [qw,qx,qy] * [gx,gy,gz]
    imu->A_data[3*NUM_STATES+0] += half_dt;   // qw/dt
    imu->A_data[3*NUM_STATES+1] += half_dt;   // qy/dt
    imu->A_data[3*NUM_STATES+2] += -half_dt;  // -qx/dt
    
    // Also update the B matrix
    memset(imu->B_data, 0, NUM_STATES*NUM_INPUTS*sizeof(matrix_data_t));
    
    // B maps how gyro measurements directly affect quaternion rate
    // qw row (row 0)
    imu->B_data[0*NUM_INPUTS+0] = -half_dt * q1;  // -qx * gx
    imu->B_data[0*NUM_INPUTS+1] = -half_dt * q2;  // -qy * gy
    imu->B_data[0*NUM_INPUTS+2] = -half_dt * q3;  // -qz * gz
    
    // qx row (row 1)
    imu->B_data[1*NUM_INPUTS+0] = half_dt * q0;   // qw * gx
    imu->B_data[1*NUM_INPUTS+1] = half_dt * q3;   // qz * gy
    imu->B_data[1*NUM_INPUTS+2] = -half_dt * q2;  // -qy * gz
    
    // qy row (row 2)
    imu->B_data[2*NUM_INPUTS+0] = -half_dt * q3;  // -qz * gx
    imu->B_data[2*NUM_INPUTS+1] = half_dt * q0;   // qw * gy
    imu->B_data[2*NUM_INPUTS+2] = half_dt * q1;   // qx * gz
    
    // qz row (row 3)
    imu->B_data[3*NUM_INPUTS+0] = half_dt * q2;   // qy * gx
    imu->B_data[3*NUM_INPUTS+1] = -half_dt * q1;  // -qx * gy
    imu->B_data[3*NUM_INPUTS+2] = half_dt * q0;   // qw * gz
    
    // Bias affects gyro directly with correct sign (negative)
    imu->B_data[4*NUM_INPUTS+0] = -1.0f;  // bias_x -> gx (negative since bias is subtracted)
    imu->B_data[5*NUM_INPUTS+1] = -1.0f;  // bias_y -> gy (negative since bias is subtracted)
    imu->B_data[6*NUM_INPUTS+2] = -1.0f;  // bias_z -> gz (negative since bias is subtracted)
}

// Update H matrix (measurement matrix) based on current quaternion
void update_H_matrix(int imu_index) {
    IMUConfig* imu = &imus[imu_index];
    
    // Extract quaternion
    float q0 = imu->x_data[0]; // w
    float q1 = imu->x_data[1]; // x
    float q2 = imu->x_data[2]; // y
    float q3 = imu->x_data[3]; // z
    
    // Clear the H matrix
    memset(imu->H_data, 0, NUM_MEASUREMENTS*NUM_STATES*sizeof(matrix_data_t));
    
    // H maps quaternion to gravity direction (partial derivatives)
    // For g_x = 2*(q1*q3 - q0*q2)
    imu->H_data[0*NUM_STATES+0] = -2.0f * q2;  // d(g_x)/d(q0)
    imu->H_data[0*NUM_STATES+1] = 2.0f * q3;   // d(g_x)/d(q1)
    imu->H_data[0*NUM_STATES+2] = -2.0f * q0;  // d(g_x)/d(q2)
    imu->H_data[0*NUM_STATES+3] = 2.0f * q1;   // d(g_x)/d(q3)
    
    // For g_y = 2*(q0*q1 + q2*q3)
    imu->H_data[1*NUM_STATES+0] = 2.0f * q1;   // d(g_y)/d(q0)
    imu->H_data[1*NUM_STATES+1] = 2.0f * q0;   // d(g_y)/d(q1)
    imu->H_data[1*NUM_STATES+2] = 2.0f * q3;   // d(g_y)/d(q2)
    imu->H_data[1*NUM_STATES+3] = 2.0f * q2;   // d(g_y)/d(q3)
    
    // For g_z = q0^2 - q1^2 - q2^2 + q3^2
    imu->H_data[2*NUM_STATES+0] = 2.0f * q0;   // d(g_z)/d(q0)
    imu->H_data[2*NUM_STATES+1] = -2.0f * q1;  // d(g_z)/d(q1)
    imu->H_data[2*NUM_STATES+2] = -2.0f * q2;  // d(g_z)/d(q2)
    imu->H_data[2*NUM_STATES+3] = 2.0f * q3;   // d(g_z)/d(q3)
}

// Normalize quaternion in state vector
void normalize_quaternion(int imu_index) {
    IMUConfig* imu = &imus[imu_index];
    
    float q0 = imu->x_data[0];
    float q1 = imu->x_data[1];
    float q2 = imu->x_data[2];
    float q3 = imu->x_data[3];
    
    float q_norm = q0*q0 + q1*q1 + q2*q2 + q3*q3;
    
    if (q_norm > 1e-10f) {  // Use a smaller threshold for better numerical stability
        q_norm = 1.0f / sqrtf(q_norm);
        imu->x_data[0] = q0 * q_norm;
        imu->x_data[1] = q1 * q_norm;
        imu->x_data[2] = q2 * q_norm;
        imu->x_data[3] = q3 * q_norm;
    } else {
        // Reset to identity if unstable
        imu->x_data[0] = 1.0f;
        imu->x_data[1] = 0.0f;
        imu->x_data[2] = 0.0f;
        imu->x_data[3] = 0.0f;
        
        if (debug_mode) {
            printf("Warning: Unstable quaternion detected, reset to identity\n");
        }
    }
}

// Update gyro bias during stationary periods
void update_gyro_bias(int imu_index) {
    IMUConfig* imu = &imus[imu_index];
    
    if (settings.auto_bias_update && is_static[imu_index]) {
        float bias_K_gain = 0.001f;
        
        // Use raw gyro values for bias update
        float raw_gx = imu->gyro[0] + imu->x_data[4];  // Add back the bias to get raw value
        float raw_gy = imu->gyro[1] + imu->x_data[5];
        float raw_gz = imu->gyro[2] + imu->x_data[6];
        
        // Update bias estimate
        imu->x_data[4] += bias_K_gain * raw_gx;
        imu->x_data[5] += bias_K_gain * raw_gy;
        imu->x_data[6] += bias_K_gain * raw_gz;
    }
}

// Extract quaternion from state vector to IMU quaternion
void extract_quaternion_from_state(int imu_index) {
    IMUConfig* imu = &imus[imu_index];
    
    float qw = imu->x_data[0];
    float qx = imu->x_data[1];
    float qy = imu->x_data[2];
    float qz = imu->x_data[3];
    
    // Verify quaternion before assigning
    float mag = qw*qw + qx*qx + qy*qy + qz*qz;
    
    if (mag > 0.1f && isfinite(qw) && isfinite(qx) && isfinite(qy) && isfinite(qz)) {
        // Normalize again to be extra safe
        float norm = 1.0f / sqrtf(mag);
        imu->quaternion[0] = qw * norm;
        imu->quaternion[1] = qx * norm;
        imu->quaternion[2] = qy * norm;
        imu->quaternion[3] = qz * norm;
    } else {
        // Reset to identity if invalid
        imu->quaternion[0] = 1.0f;
        imu->quaternion[1] = 0.0f;
        imu->quaternion[2] = 0.0f;
        imu->quaternion[3] = 0.0f;
        
        if (debug_mode) {
            printf("Warning: Invalid quaternion detected during extraction\n");
        }
    }
}

// Reset gyro bias values
void reset_gyro_bias() {
    for (int i = 0; i < imu_count; i++) {
        // Reset bias part of state vector
        imus[i].x_data[4] = 0.0f;
        imus[i].x_data[5] = 0.0f;
        imus[i].x_data[6] = 0.0f;
    }
    if (debug_mode) {
        printf("Gyro bias reset for all IMUs\n");
    }
}

// Update all IMU quaternions using Kalman filter
void update_quaternions_kalman() {
    for (int i = 0; i < imu_count; i++) {
        IMUConfig* imu = &imus[i];
        
        // 1. Update state transition matrix based on current quaternion
        update_state_transition(i, last_dt);
        
        // 2. Update input vector from gyro measurements (in rad/s)
        imu->u_data[0] = imu->gyro[0] * (M_PI / 180.0f);
        imu->u_data[1] = imu->gyro[1] * (M_PI / 180.0f);
        imu->u_data[2] = imu->gyro[2] * (M_PI / 180.0f);
        
        // 3. Kalman filter prediction steps using standard library
        kalman_predict_x(&imu->kf);
        kalman_predict_Q(&imu->kf);
        
        // 4. Normalize quaternion part of state
        normalize_quaternion(i);
        
        // 5. Check if accelerometer reading is valid
        float acc_mag = sqrtf(imu->accel[0]*imu->accel[0] + 
                              imu->accel[1]*imu->accel[1] + 
                              imu->accel[2]*imu->accel[2]);
        
        // Only use accelerometer for update if magnitude is close to 1g
        if (acc_mag > 0.5f && acc_mag < 1.5f) {
            // 6. Update measurement matrix based on current quaternion
            update_H_matrix(i);
            
            // 7. Normalize accelerometer reading for measurement
            imu->z_data[0] = -imu->accel[0] / acc_mag;
            imu->z_data[1] = -imu->accel[1] / acc_mag;
            imu->z_data[2] = -imu->accel[2] / acc_mag;
            
            // 8. Calculate expected gravity direction from current quaternion
            float q0 = imu->x_data[0]; // w
            float q1 = imu->x_data[1]; // x
            float q2 = imu->x_data[2]; // y
            float q3 = imu->x_data[3]; // z
            
            // Expected gravity direction
            float expected_g[3];
            expected_g[0] = 2.0f * (q1*q3 - q0*q2);
            expected_g[1] = 2.0f * (q0*q1 + q2*q3);
            expected_g[2] = 2.0f * (0.5f*q0*q0 - 0.5f*q1*q1 - 0.5f*q2*q2 + 0.5f*q3*q3);
            
            // 9. Calculate innovation: y = z - h(x)
            for (int j = 0; j < 3; j++) {
                imu->y_data[j] = imu->z_data[j] - expected_g[j];
            }
            
            // 10. Kalman filter correction step using standard library
            kalman_correct(&imu->kf, &imu->kfm);
            
            // 11. Normalize quaternion after update
            normalize_quaternion(i);
        }
        
        // 12. Update gyro bias during stationary periods
        update_gyro_bias(i);
        
        // 13. Extract quaternion from state
        extract_quaternion_from_state(i);
    }
}

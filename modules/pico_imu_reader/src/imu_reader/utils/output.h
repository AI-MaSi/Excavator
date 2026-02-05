#ifndef OUTPUT_H
#define OUTPUT_H

#include "imu_reader.h"
#include <inttypes.h>

#define NUM_SENSORS 3
#define FLOATS_PER_SENSOR 7

// Message types for control frames (version=2)
#define MSG_TYPE_CFG_OK    0x01
#define MSG_TYPE_CFG_WAIT  0x02
#define MSG_TYPE_ZERO_ACK  0x03
#define MSG_TYPE_ERROR     0x04

void print_output_data(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR]);
void send_control_msg(uint8_t msg_type);

extern imu_reader_settings_t imu_reader_settings;

#endif

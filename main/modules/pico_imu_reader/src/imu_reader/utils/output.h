#ifndef OUTPUT_H
#define OUTPUT_H
#include "imu_reader.h"
#include <inttypes.h>
void print_output_data (uint64_t ts_us, float sensors_data[][11]);
extern imu_reader_settings_t imu_reader_settings;
#endif

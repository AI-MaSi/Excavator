#ifndef ISM330DLC_H
#define ISM330DLC_H
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "hardware/i2c.h"
#include "ism330dlc_registers.h"
#include "ism330dlc_config.h"
#include "bit_ops.h"
#include "FusionMath.h"
#include "FusionStructs.h"
#include "imu_reader.h"

bool ism330dhcx_write_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t value);
bool ism330dhcx_read_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value, uint8_t read_count);
void print_list(uint8_t list[], int size);
bool ism330dhcx_wait_for_data(i2c_inst_t* i2c_port, uint8_t device_addr, uint32_t timeout_us);
bool ism330dhcx_read_gyro(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector);
bool ism330dhcx_read_accelerometer(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector);
bool ism330dhcx_read(i2c_inst_t* i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value);
bool ism330dhcx_init(i2c_inst_t *i2c_port, uint8_t device_addr, uint internal_odr_hz);
int initialize_sensors(uint internal_odr_hz);
uint8_t get_active_sensor_count(void);
uint8_t get_active_sensor_bus(uint8_t sensor_index);
uint8_t get_active_sensor_addr(uint8_t sensor_index);
bool read_active_sensor_motion_unbiased(uint8_t sensor_index, FusionVector* accelerometer, FusionVector* gyroscope);
void set_active_sensor_gyro_bias(uint8_t sensor_index, FusionVector gyro_bias);
void read_all_sensors(Sensor* sensors, bool sensor_valid[]);
#endif

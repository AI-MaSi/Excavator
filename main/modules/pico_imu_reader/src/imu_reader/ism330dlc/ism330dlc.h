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
bool ism330dhcx_read_gyro(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector);
bool ism330dhcx_read_accelerometer(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector);
bool ism330dhcx_read(i2c_inst_t* i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value);
bool ism330dhcx_init(i2c_inst_t *i2c_port, uint8_t device_addr);
int initialize_sensors(void);
void read_all_sensors(Sensor* sensors);
#endif

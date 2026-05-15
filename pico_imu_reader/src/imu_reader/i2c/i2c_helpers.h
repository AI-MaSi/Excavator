#ifndef I2C_HELPERS_H
#define I2C_HELPERS_H
#include <stdio.h>         
#include <pico/stdlib.h>   
#include <hardware/i2c.h>
#include <pico/error.h>
#include "imu_reader.h"
bool reserved_addr(uint8_t addr);
int setup_I2C_pins(uint baud_rate);
void i2c_scan(i2c_inst_t *i2c_port);
// Fixed Seeed XIAO RP2040 IMU wiring:
// - I2C0 uses P28/P29
// - I2C1 uses P6/P7
// Do not change these without updating the hardware wiring as well.
#define I2C_PORT_0 i2c0
#define I2C_SDA_0 28
#define I2C_SCL_0 29

// Fixed Seeed XIAO RP2040 IMU wiring for second bus: P6/P7
#define I2C_PORT_1 i2c1
#define I2C_SDA_1 6
#define I2C_SCL_1 7

#endif

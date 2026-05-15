#ifndef STATUS_LED_H
#define STATUS_LED_H

#include <stdint.h>

// Status LED colors for XIAO RP2040 onboard NeoPixel (WS2812)
// GPIO 12 = data, GPIO 11 = power
#define STATUS_LED_PIN      12
#define STATUS_LED_POWER_PIN 11

typedef enum {
    STATUS_BOOT      = 0,   // Red     — just booted, waiting for USB
    STATUS_CFG_WAIT  = 1,   // Yellow  — USB connected, waiting for config
    STATUS_INIT      = 2,   // Blue    — initializing I2C / IMU
    STATUS_STREAM    = 3,   // Green   — streaming data
    STATUS_ERROR     = 4,   // Red flash — error
    STATUS_CALIBRATE = 5,   // Orange  — stationary gyro-bias calibration
} status_led_state_t;

void status_led_init(void);
void status_led_set(status_led_state_t state);
void status_led_set_rgb(uint8_t r, uint8_t g, uint8_t b);
void status_led_off(void);

#endif // STATUS_LED_H

#include "status_led.h"
#include "hardware/pio.h"
#include "hardware/gpio.h"
#include "ws2812.pio.h"

// Use PIO0, state machine 0
static PIO pio = pio0;
static uint sm = 0;
static bool initialized = false;

static inline void put_pixel(uint32_t grb) {
    pio_sm_put_blocking(pio, sm, grb << 8u);
}

static inline uint32_t urgb_u32(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)g << 16) | ((uint32_t)r << 8) | (uint32_t)b;
}

void status_led_init(void) {
    // Power on the NeoPixel (XIAO RP2040 has a power gate on GPIO 11)
    gpio_init(STATUS_LED_POWER_PIN);
    gpio_set_dir(STATUS_LED_POWER_PIN, GPIO_OUT);
    gpio_put(STATUS_LED_POWER_PIN, 1);

    // Init PIO for WS2812
    uint offset = pio_add_program(pio, &ws2812_program);
    ws2812_program_init(pio, sm, offset, STATUS_LED_PIN, 800000, false);
    initialized = true;

    // Start off
    put_pixel(0);
}

void status_led_set_rgb(uint8_t r, uint8_t g, uint8_t b) {
    if (!initialized) return;
    put_pixel(urgb_u32(r, g, b));
}

void status_led_off(void) {
    status_led_set_rgb(0, 0, 0);
}

void status_led_set(status_led_state_t state) {
    // Minimal brightness — still clearly visible
    switch (state) {
        case STATUS_BOOT:       status_led_set_rgb( 3,  1,  0); break;  // Amber (reddish)
        case STATUS_CFG_WAIT:   status_led_set_rgb( 2,  2,  0); break;  // Yellow (greenish)
        case STATUS_INIT:       status_led_set_rgb( 0,  0,  3); break;  // Blue
        case STATUS_STREAM:     status_led_set_rgb( 0,  3,  0); break;  // Green
        case STATUS_ERROR:      status_led_set_rgb( 5,  0,  0); break;  // Red
        case STATUS_CALIBRATE:  status_led_set_rgb(24,  8,  0); break;  // Bright orange
        default:                status_led_off();                break;
    }
}

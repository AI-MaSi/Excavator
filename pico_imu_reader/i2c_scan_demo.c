#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/gpio.h"
#include "hardware/i2c.h"

#define I2C_BAUD_HZ 100000
#define WHO_AM_I_REG 0x0F
#define ISM330DHCX_ID 0x6B

typedef struct ScanTarget {
    const char *label;
    i2c_inst_t *inst;
    uint sda;
    uint scl;
} ScanTarget;

static bool reserved_addr(uint8_t addr) {
    return (addr & 0x78u) == 0 || (addr & 0x78u) == 0x78u;
}

static void reset_pins(uint sda, uint scl) {
    gpio_deinit(sda);
    gpio_deinit(scl);
}

static void setup_bus(i2c_inst_t *inst, uint sda, uint scl) {
    i2c_init(inst, I2C_BAUD_HZ);
    gpio_set_function(sda, GPIO_FUNC_I2C);
    gpio_set_function(scl, GPIO_FUNC_I2C);
    gpio_pull_up(sda);
    gpio_pull_up(scl);
}

static bool read_who_am_i(i2c_inst_t *inst, uint8_t addr, uint8_t *value) {
    uint8_t reg = WHO_AM_I_REG;
    int wr = i2c_write_timeout_us(inst, addr, &reg, 1, true, 2000);
    if (wr != 1) {
        return false;
    }
    int rd = i2c_read_timeout_us(inst, addr, value, 1, false, 2000);
    return rd == 1;
}

static void scan_target(const ScanTarget *target) {
    printf("\n[%s] using %s on GPIO%u/ GPIO%u\n",
           target->label,
           target->inst == i2c0 ? "i2c0" : "i2c1",
           target->sda,
           target->scl);

    setup_bus(target->inst, target->sda, target->scl);

    int found = 0;
    printf("  ACK addresses:");
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        if (reserved_addr(addr)) {
            continue;
        }

        uint8_t rx = 0;
        int ret = i2c_read_timeout_us(target->inst, addr, &rx, 1, false, 2000);
        if (ret >= 0) {
            printf(" 0x%02X", addr);
            found++;
        }
    }
    if (found == 0) {
        printf(" none");
    }
    printf("\n");

    for (uint8_t addr = 0x6A; addr <= 0x6B; addr++) {
        uint8_t who = 0;
        if (read_who_am_i(target->inst, addr, &who)) {
            printf("  WHO_AM_I 0x%02X -> 0x%02X", addr, who);
            if (who == ISM330DHCX_ID) {
                printf(" (ISM330DHCX)\n");
            } else {
                printf("\n");
            }
        } else {
            printf("  WHO_AM_I 0x%02X -> no response\n", addr);
        }
    }

    i2c_deinit(target->inst);
    reset_pins(target->sda, target->scl);
}

int main(void) {
    stdio_init_all();

    for (int i = 0; i < 50 && !stdio_usb_connected(); i++) {
        sleep_ms(100);
    }
    sleep_ms(250);

    const ScanTarget targets[] = {
        {.label = "P28/P29", .inst = i2c0, .sda = 28, .scl = 29},
        {.label = "P26/P27", .inst = i2c1, .sda = 26, .scl = 27},
        {.label = "P6/P7", .inst = i2c1, .sda = 6, .scl = 7},
    };

    printf("\nI2C scan demo for Seeed XIAO RP2040\n");
    printf("Scanning P28/P29, P26/P27, and P6/P7 once per second.\n");

    uint32_t iteration = 0;
    while (true) {
        printf("\n=== Scan %lu ===\n", (unsigned long)iteration++);
        for (size_t i = 0; i < sizeof(targets) / sizeof(targets[0]); i++) {
            scan_target(&targets[i]);
        }
        fflush(stdout);
        sleep_ms(1000);
    }
}

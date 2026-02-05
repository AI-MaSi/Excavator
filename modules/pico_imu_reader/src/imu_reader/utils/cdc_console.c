#include "cdc_console.h"
#include "tusb.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define CDC_CONSOLE_BUF_LEN 256

static volatile bool cdc_console_enabled = true;

void cdc_console_enable(bool enabled) {
    cdc_console_enabled = enabled;
}

bool cdc_console_is_enabled(void) {
    return cdc_console_enabled;
}

void cdc_write_str(const char *s) {
    if (!cdc_console_enabled || s == NULL) {
        return;
    }
    if (!tud_cdc_connected()) {
        return;
    }
    size_t len = strlen(s);
    if (len == 0) {
        return;
    }
    tud_cdc_write(s, len);
    tud_cdc_write_flush();
}

void cdc_write_line(const char *s) {
    if (!cdc_console_enabled) {
        return;
    }
    if (s != NULL) {
        cdc_write_str(s);
    }
    cdc_write_str("\n");
}

void cdc_writef(const char *fmt, ...) {
    if (!cdc_console_enabled || fmt == NULL) {
        return;
    }
    if (!tud_cdc_connected()) {
        return;
    }
    char buf[CDC_CONSOLE_BUF_LEN];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (n <= 0) {
        return;
    }
    size_t len = (n < (int)sizeof(buf)) ? (size_t)n : (sizeof(buf) - 1);
    tud_cdc_write(buf, len);
    tud_cdc_write_flush();
}

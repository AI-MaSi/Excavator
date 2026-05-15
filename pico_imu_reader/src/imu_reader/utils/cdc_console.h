#ifndef CDC_CONSOLE_H
#define CDC_CONSOLE_H

#include <stdbool.h>

void cdc_console_enable(bool enabled);
bool cdc_console_is_enabled(void);
void cdc_write_str(const char *s);
void cdc_write_line(const char *s);
void cdc_writef(const char *fmt, ...);

#endif // CDC_CONSOLE_H

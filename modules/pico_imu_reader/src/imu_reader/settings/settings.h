#ifndef SETTINGS_H
#define SETTINGS_H

#include "imu_reader.h"

#define SETTINGS_BUF_LEN (256)

void wait_for_settings(void);
void check_for_zero_command(void);

extern volatile int zero_requested;

#endif //SETTINGS_H

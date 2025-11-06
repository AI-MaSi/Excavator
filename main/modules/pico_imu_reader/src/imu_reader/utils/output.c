#include "output.h"
#include <stdio.h>

void print_output_data (uint64_t ts_us, float sensors_data[][11]) {
    // Prepend timestamp (microseconds since boot), then per sensor -> w,x,y,z,gx,gy,gz
    printf("%" PRIu64, ts_us);
    for (int i = 0; i < imu_reader_settings.sensorCount; i++) {
        printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
               sensors_data[i][0], sensors_data[i][1], sensors_data[i][2], sensors_data[i][3],
               sensors_data[i][4], sensors_data[i][5], sensors_data[i][6]);
    }
    printf("\n");
}

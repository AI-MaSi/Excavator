#ifndef INITFUSION_H
#define INITFUSION_H

#include "Fusion.h"
#include <time.h>
#include "FusionStructs.h"
#include "i2c_helpers.h"
#include "imu_reader.h"

void initialize_sensors_values(Sensor* sensors, int count);
void initialize_calibrations(Sensor* sensors, int count);
void initialize_algos(Sensor* sensors, int count);

#endif //INITFUSION_H

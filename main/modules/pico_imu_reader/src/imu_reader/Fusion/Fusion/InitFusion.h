#ifndef INITFUSION_H
#define INITFUSION_H

#include "Fusion.h"
#include <time.h>
#include "FusionStructs.h"
#include "i2c_helpers.h"
#include "imu_reader.h"

void initialize_calibrations(Sensor* sensors);
void initialize_algos(Sensor* sensors);
void initialize_sensors_values(Sensor* sensors);
#endif //INITFUSION_H



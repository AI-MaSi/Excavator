#include "ism330dlc.h"
#include "i2c_helpers.h"
#include "FusionMath.h"
#include "pico/time.h"

// Function to write to ISM330DHCX register
bool ism330dhcx_write_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    int result = i2c_write_blocking(i2c_port, device_addr, buf, 2, false);
    return result == 2;
}

// Function to read from ISM330DHCX register
bool ism330dhcx_read_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value, uint8_t read_count) {
    int result = i2c_write_blocking(i2c_port, device_addr, &reg, 1, true);
    if (result != 1) return false;
    result = i2c_read_blocking(i2c_port, device_addr, value, read_count, false);
    return result == 1;
}


void print_list(uint8_t list[], int size){
    printf("Raw gyro values: ");
    for (int i = 0; i < size; i++){
        printf("%d ", list[i]);
    }
    printf("\n");
}

bool ism330dhcx_read_gyro(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector) {
    uint8_t raw_gyro_values[6];
	ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_G ,raw_gyro_values ,6);
    int16_t raw_gyro_x = combine_8_bits(raw_gyro_values[0], raw_gyro_values[1]);
    int16_t raw_gyro_y = combine_8_bits(raw_gyro_values[2], raw_gyro_values[3]);
    int16_t raw_gyro_z = combine_8_bits(raw_gyro_values[4], raw_gyro_values[5]);

    fusion_vector->axis.x = ((float)raw_gyro_x/32768.0f)*(float)G_DPS_RANGE;
    fusion_vector->axis.y = ((float)raw_gyro_y/32768.0f)*(float)G_DPS_RANGE;
    fusion_vector->axis.z = ((float)raw_gyro_z/32768.0f)*(float)G_DPS_RANGE;
	return 1;
}

bool ism330dhcx_read(i2c_inst_t* i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value) {
	// ism330dhcx_read_accelerometer();
	// ism330dhcx_read_gyro();
	return 0;
}

// Choose ODR field for requested sample rate; returns CTRL ODR bits in [7:4]
static inline uint8_t select_odr_bits(int requested_hz) {
    if (requested_hz <= 104) return 0x40;   // 104 Hz
    if (requested_hz <= 208) return 0x50;   // 208 Hz
    if (requested_hz <= 416) return 0x60;   // 416 Hz
    if (requested_hz <= 833) return 0x70;   // 833 Hz
    // Fallback to 1.66 kHz for anything higher
    return 0x80;                             // 1660 Hz
}

// Initialize ISM330DHCX
bool ism330dhcx_init(i2c_inst_t *i2c_port, uint8_t device_addr) {
    // Optional: enable auto-increment + BDU for clean multi-byte reads
    const uint8_t CTRL3_C_IF_INC = 0x04;   // bit2
    const uint8_t CTRL3_C_BDU    = 0x40;   // bit6
    (void)ism330dhcx_write_reg(i2c_port, device_addr, CTRL3_C, CTRL3_C_IF_INC | CTRL3_C_BDU);

    // Map requested sample rate to nearest supported ODR at or above request
    const uint8_t odr_bits = select_odr_bits(imu_reader_settings.sampleRate);

    // Configure accelerometer: ODR + ±2g
    const uint8_t xl_cntrl1_val = odr_bits | XL_G_RANGE_MASK;
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL1_XL, xl_cntrl1_val)) {
        printf("Failed to configure accelerometer (addr=0x%02x)\n", device_addr);
        return false;
    }

    // Configure gyroscope: ODR + ±250 dps
    const uint8_t g_cntrl_val = odr_bits | G_DPS_RANGE_MASK;
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL2_G, g_cntrl_val)) {
        printf("Failed to configure gyroscope (addr=0x%02x)\n", device_addr);
        return false;
    }
    // Enable in-sensor LPF to reduce vibration/aliasing while keeping responsiveness
    // Accelerometer LPF2: enable and choose a moderate cutoff via HPCF_XL[1:0]
    // Read-modify-write CTRL8_XL (0x17): bit7 LPF2_XL_EN, bits1:0 HPCF_XL
    {
        uint8_t ctrl8 = 0;
        (void)ism330dhcx_read_reg(i2c_port, device_addr, CTRL8_XL, &ctrl8, 1);
        ctrl8 |= 0x80;               // LPF2_XL_EN = 1
        ctrl8 &= (uint8_t)~0x03;     // clear HPCF_XL
        ctrl8 |= 0x01;               // HPCF_XL = 01b (moderate cutoff)
        (void)ism330dhcx_write_reg(i2c_port, device_addr, CTRL8_XL, ctrl8);
    }

    // Gyroscope LPF1 bandwidth: set FTYPE[3:0] in CTRL6_C (0x15) to mid cutoff
    {
        uint8_t ctrl6 = 0;
        (void)ism330dhcx_read_reg(i2c_port, device_addr, CTRL6_C, &ctrl6, 1);
        ctrl6 &= (uint8_t)~0x0F;     // clear FTYPE[3:0]
        ctrl6 |= 0x03;               // FTYPE = 0x3 (mid bandwidth)
        (void)ism330dhcx_write_reg(i2c_port, device_addr, CTRL6_C, ctrl6);
    }

    printf("ISM330: accel LPF2 enabled (HPCF=1); gyro LPF1 FTYPE=3\n");

    return true;
}

int initialize_sensors(void) {
	printf("Intializing sensors\n");
    // Initialize sensors
    if (imu_reader_settings.channelCount == 2) {
        printf("Intializing both i2c channels!\n");
        if (!ism330dhcx_init(I2C_PORT_0, ISM330DHCX_ADDR_DO_LOW)) {
            printf("Failed to initialize ISM330DHCX with i2c_port: %s and i2c_address: 0x%02x!\n", I2C_PORT_0, ISM330DHCX_ADDR_DO_LOW);
        }
    
       if (!ism330dhcx_init(I2C_PORT_0, ISM330DHCX_ADDR_DO_HIGH)) {
                printf("Failed to initialize ISM330DHCX with i2c_port: %s and i2c_address: 0x%02x!\n", I2C_PORT_0, ISM330DHCX_ADDR_DO_HIGH);
          }
        if (!ism330dhcx_init(I2C_PORT_1, ISM330DHCX_ADDR_DO_LOW)) {
            printf("Failed to initialize ISM330DHCX with i2c_port: %s and i2c_address: 0x%02x!\n", I2C_PORT_1, ISM330DHCX_ADDR_DO_LOW);
        }
    } else {
        printf("Initializing only i2c channel 0!\n");
        if (!ism330dhcx_init(I2C_PORT_0, ISM330DHCX_ADDR_DO_LOW)) {
            printf("Failed to initialize ISM330DHCX with i2c_port: %s and i2c_address: 0x%02x!\n", I2C_PORT_0, ISM330DHCX_ADDR_DO_LOW);
        }
    
       if (!ism330dhcx_init(I2C_PORT_0, ISM330DHCX_ADDR_DO_HIGH)) {
                printf("Failed to initialize ISM330DHCX with i2c_port: %s and i2c_address: 0x%02x!\n", I2C_PORT_0, ISM330DHCX_ADDR_DO_HIGH);
          }
    }
    printf("Sensor initialized successfully!\n");
}

// Wait for new data to be ready from the sensor
// Bit 0: XLDA (accelerometer data available)
// Bit 1: GDA (gyroscope data available)
bool ism330dhcx_wait_for_data(i2c_inst_t* i2c_port, uint8_t device_addr, uint32_t timeout_us) {
    uint64_t start = time_us_64();
    uint8_t status;

    while ((time_us_64() - start) < timeout_us) {
        if (ism330dhcx_read_reg(i2c_port, device_addr, STATUS_REG, &status, 1)) {
            // Check if both accelerometer (bit 0) and gyroscope (bit 1) data are ready
            if ((status & 0x03) == 0x03) {
                return true;
            }
        }
        // Small delay to avoid hammering the I2C bus
        sleep_us(10);
    }
    return false; // Timeout
}

bool ism330dhcx_read_accelerometer(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector) {
    uint8_t raw_acc_values[6];
	ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_XL ,raw_acc_values ,6);
    int16_t raw_acc_x = combine_8_bits(raw_acc_values[0], raw_acc_values[1]);
    int16_t raw_acc_y = combine_8_bits(raw_acc_values[2], raw_acc_values[3]);
    int16_t raw_acc_z = combine_8_bits(raw_acc_values[4], raw_acc_values[5]);

    fusion_vector->axis.x = ((float)raw_acc_x/32768.0f)*(float)XL_G_RANGE;
    fusion_vector->axis.y = ((float)raw_acc_y/32768.0f)*(float)XL_G_RANGE;
    fusion_vector->axis.z = ((float)raw_acc_z/32768.0f)*(float)XL_G_RANGE;
	return 1;
}

void read_all_sensors(Sensor* sensors) {
    int index = 0;
   for (int i = 0; i < imu_reader_settings.channelCount; i++) {
        if(i == 0) {
            if (imu_reader_settings.sensorCount >= 2) {
                ism330dhcx_read_accelerometer(I2C_PORT_0,ISM330DHCX_ADDR_DO_HIGH, &sensors[index].accelerometer);
                ism330dhcx_read_gyro(I2C_PORT_0,ISM330DHCX_ADDR_DO_HIGH, &sensors[index].gyroscope);
                sensors[index].timestamp = time_us_64();
                index++;
                ism330dhcx_read_accelerometer(I2C_PORT_0,ISM330DHCX_ADDR_DO_LOW, &sensors[index].accelerometer);
                ism330dhcx_read_gyro(I2C_PORT_0,ISM330DHCX_ADDR_DO_LOW, &sensors[index].gyroscope);
                sensors[index].timestamp = time_us_64();
                index++;
            } else { // one sensor only
                ism330dhcx_read_accelerometer(I2C_PORT_0,ISM330DHCX_ADDR_DO_HIGH, &sensors[index].accelerometer);
                ism330dhcx_read_gyro(I2C_PORT_0,ISM330DHCX_ADDR_DO_HIGH, &sensors[index].gyroscope);
                sensors[index].timestamp = time_us_64();
            }
        } else if (i == 1) {
            ism330dhcx_read_accelerometer(I2C_PORT_1,ISM330DHCX_ADDR_DO_LOW, &sensors[index].accelerometer);
            ism330dhcx_read_gyro(I2C_PORT_1,ISM330DHCX_ADDR_DO_LOW, &sensors[index].gyroscope);
            sensors[index].timestamp = time_us_64();
            index++;
        }
   }
}
 

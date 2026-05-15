#include "ism330dlc.h"
#include "i2c_helpers.h"
#include "output.h"
#include "FusionMath.h"
#include "cdc_console.h"
#include "pico/time.h"
#include <math.h>

#define I2C_REG_TIMEOUT_US 2000u

typedef struct ActiveImu {
    i2c_inst_t *i2c_port;
    uint8_t bus_index;
    uint8_t device_addr;
} ActiveImu;

static ActiveImu active_imus[MAX_SENSORS];
static FusionVector active_gyro_biases[MAX_SENSORS];
static uint8_t active_sensor_count = 0;

static bool ism330dhcx_probe(i2c_inst_t *i2c_port, uint8_t device_addr) {
    uint8_t who_am_i = 0;
    if (!ism330dhcx_read_reg(i2c_port, device_addr, WHO_AM_I, &who_am_i, 1)) {
        return false;
    }
    return who_am_i == ISM330DHCX_ID;
}

// Function to write to ISM330DHCX register
bool ism330dhcx_write_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    int result = i2c_write_timeout_us(i2c_port, device_addr, buf, 2, false, I2C_REG_TIMEOUT_US);
    return result == 2;
}

// Function to read from ISM330DHCX register
bool ism330dhcx_read_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value, uint8_t read_count) {
    int result = i2c_write_timeout_us(i2c_port, device_addr, &reg, 1, true, I2C_REG_TIMEOUT_US);
    if (result != 1) return false;
    result = i2c_read_timeout_us(i2c_port, device_addr, value, read_count, false, I2C_REG_TIMEOUT_US);
    return result == read_count;
}


void print_list(uint8_t list[], int size){
    cdc_write_str("Raw gyro values: ");
    for (int i = 0; i < size; i++){
        cdc_writef("%d ", list[i]);
    }
    cdc_write_str("\n");
}

bool ism330dhcx_read_gyro(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector) {
    uint8_t raw_gyro_values[6];
    if (!ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_G, raw_gyro_values, 6)) {
        return false;
    }
    int16_t raw_gyro_x = combine_8_bits(raw_gyro_values[0], raw_gyro_values[1]);
    int16_t raw_gyro_y = combine_8_bits(raw_gyro_values[2], raw_gyro_values[3]);
    int16_t raw_gyro_z = combine_8_bits(raw_gyro_values[4], raw_gyro_values[5]);

    const float range = (imu_reader_settings.gyroRangeDps > 0.0f) ? imu_reader_settings.gyroRangeDps : 500.0f;
    fusion_vector->axis.x = ((float)raw_gyro_x/32768.0f)*range;
    fusion_vector->axis.y = ((float)raw_gyro_y/32768.0f)*range;
    fusion_vector->axis.z = ((float)raw_gyro_z/32768.0f)*range;
    return true;
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

// Map gyro range (dps) to CTRL2_G FS_G/FS_125 bits.
static inline uint8_t gyro_range_mask(float dps) {
    if (dps <= 125.0f) return 0x02;  // FS_125 = 1
    if (dps <= 250.0f) return 0x00;  // FS_G = 00
    if (dps <= 500.0f) return 0x04;  // FS_G = 01
    if (dps <= 1000.0f) return 0x08; // FS_G = 10
    return 0x0C;                     // FS_G = 11 (2000 dps)
}

// Initialize ISM330DHCX
bool ism330dhcx_init(i2c_inst_t *i2c_port, uint8_t device_addr, uint internal_odr_hz) {
    // Optional: enable auto-increment + BDU for clean multi-byte reads
    const uint8_t CTRL3_C_IF_INC = 0x04;   // bit2
    const uint8_t CTRL3_C_BDU    = 0x40;   // bit6
    (void)ism330dhcx_write_reg(i2c_port, device_addr, CTRL3_C, CTRL3_C_IF_INC | CTRL3_C_BDU);

    // Map requested internal ODR to nearest supported hardware ODR at or above request
    const uint8_t odr_bits = select_odr_bits(internal_odr_hz);

    // Configure accelerometer: ODR + ±2g
    const uint8_t xl_cntrl1_val = odr_bits | XL_G_RANGE_MASK;
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL1_XL, xl_cntrl1_val)) {
        cdc_writef("Failed to configure accelerometer (addr=0x%02x)\n", device_addr);
        return false;
    }

    // Configure gyroscope: ODR + ±250 dps
    const float range = (imu_reader_settings.gyroRangeDps > 0.0f) ? imu_reader_settings.gyroRangeDps : 500.0f;
    const uint8_t g_cntrl_val = odr_bits | gyro_range_mask(range);
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL2_G, g_cntrl_val)) {
        cdc_writef("Failed to configure gyroscope (addr=0x%02x)\n", device_addr);
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
        // FTYPE=5 (~48 Hz) gave best quat noise at 416 Hz ODR across all tests.
        // FTYPE=7 (~12 Hz) caused phase lag on gravity axis; FTYPE=1 let in excess jitter.
        ctrl6 |= 0x05;               // FTYPE = 0x5 (~48 Hz at 416 Hz ODR)
        (void)ism330dhcx_write_reg(i2c_port, device_addr, CTRL6_C, ctrl6);
    }

    cdc_write_line("ISM330: accel LPF2 enabled (HPCF=1); gyro LPF1 FTYPE=5 (~48 Hz)");

    return true;
}

int initialize_sensors(uint internal_odr_hz) {
    static const struct {
        i2c_inst_t *i2c_port;
        uint8_t bus_index;
        uint8_t device_addr;
    } candidates[MAX_SENSORS] = {
        // Must match pico_imu_simulator and control_config.yaml imu_mapping:
        // [0] base=I2C1:0x6B, [1] boom=I2C0:0x6A,
        // [2] bucket=I2C0:0x6B, [3] arm=I2C1:0x6A.
        {I2C_PORT_1, 1, ISM330DHCX_ADDR_DO_HIGH},
        {I2C_PORT_0, 0, ISM330DHCX_ADDR_DO_LOW},
        {I2C_PORT_0, 0, ISM330DHCX_ADDR_DO_HIGH},
        {I2C_PORT_1, 1, ISM330DHCX_ADDR_DO_LOW},
    };

    active_sensor_count = 0;
    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        active_gyro_biases[i] = FUSION_VECTOR_ZERO;
    }
    cdc_write_line("Probing IMUs on I2C0/I2C1 @ 0x6A/0x6B...");

    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        const uint8_t addr = candidates[i].device_addr;
        const uint8_t bus = candidates[i].bus_index;

        if (!ism330dhcx_probe(candidates[i].i2c_port, addr)) {
            cdc_writef("No IMU at I2C%d @ 0x%02x\n", bus, addr);
            continue;
        }

        if (!ism330dhcx_init(candidates[i].i2c_port, addr, internal_odr_hz)) {
            cdc_writef("Probe OK but init failed: I2C%d @ 0x%02x\n", bus, addr);
            continue;
        }

        active_imus[active_sensor_count].i2c_port = candidates[i].i2c_port;
        active_imus[active_sensor_count].bus_index = bus;
        active_imus[active_sensor_count].device_addr = addr;
        active_sensor_count++;
        cdc_writef("Active IMU %d: I2C%d @ 0x%02x\n", active_sensor_count - 1, bus, addr);
    }

    cdc_writef("Detected %d IMU(s)\n", active_sensor_count);
    return active_sensor_count > 0 ? 1 : 0;
}

uint8_t get_active_sensor_count(void) {
    return active_sensor_count;
}

uint8_t get_active_sensor_bus(uint8_t sensor_index) {
    if (sensor_index >= active_sensor_count) {
        return 0xFF;
    }
    return active_imus[sensor_index].bus_index;
}

uint8_t get_active_sensor_addr(uint8_t sensor_index) {
    if (sensor_index >= active_sensor_count) {
        return 0xFF;
    }
    return active_imus[sensor_index].device_addr;
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
    if (!ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_XL, raw_acc_values, 6)) {
        return false;
    }
    int16_t raw_acc_x = combine_8_bits(raw_acc_values[0], raw_acc_values[1]);
    int16_t raw_acc_y = combine_8_bits(raw_acc_values[2], raw_acc_values[3]);
    int16_t raw_acc_z = combine_8_bits(raw_acc_values[4], raw_acc_values[5]);

    fusion_vector->axis.x = ((float)raw_acc_x/32768.0f)*(float)XL_G_RANGE;
    fusion_vector->axis.y = ((float)raw_acc_y/32768.0f)*(float)XL_G_RANGE;
    fusion_vector->axis.z = ((float)raw_acc_z/32768.0f)*(float)XL_G_RANGE;
    return true;
}

static bool ism330dhcx_read_motion(i2c_inst_t* i2c_port, uint8_t device_addr,
                                   FusionVector* accelerometer, FusionVector* gyroscope) {
    uint8_t raw_values[12];
    if (!ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_G, raw_values, sizeof(raw_values))) {
        return false;
    }

    const int16_t raw_gyro_x = combine_8_bits(raw_values[0], raw_values[1]);
    const int16_t raw_gyro_y = combine_8_bits(raw_values[2], raw_values[3]);
    const int16_t raw_gyro_z = combine_8_bits(raw_values[4], raw_values[5]);
    const int16_t raw_acc_x = combine_8_bits(raw_values[6], raw_values[7]);
    const int16_t raw_acc_y = combine_8_bits(raw_values[8], raw_values[9]);
    const int16_t raw_acc_z = combine_8_bits(raw_values[10], raw_values[11]);

    const float gyro_range = (imu_reader_settings.gyroRangeDps > 0.0f) ? imu_reader_settings.gyroRangeDps : 500.0f;
    gyroscope->axis.x = ((float)raw_gyro_x / 32768.0f) * gyro_range;
    gyroscope->axis.y = ((float)raw_gyro_y / 32768.0f) * gyro_range;
    gyroscope->axis.z = ((float)raw_gyro_z / 32768.0f) * gyro_range;

    accelerometer->axis.x = ((float)raw_acc_x / 32768.0f) * (float)XL_G_RANGE;
    accelerometer->axis.y = ((float)raw_acc_y / 32768.0f) * (float)XL_G_RANGE;
    accelerometer->axis.z = ((float)raw_acc_z / 32768.0f) * (float)XL_G_RANGE;
    return true;
}

bool read_active_sensor_motion_unbiased(uint8_t sensor_index, FusionVector* accelerometer, FusionVector* gyroscope) {
    if ((sensor_index >= active_sensor_count) || (accelerometer == NULL) || (gyroscope == NULL)) {
        return false;
    }

    return ism330dhcx_read_motion(
        active_imus[sensor_index].i2c_port,
        active_imus[sensor_index].device_addr,
        accelerometer,
        gyroscope);
}

void set_active_sensor_gyro_bias(uint8_t sensor_index, FusionVector gyro_bias) {
    if (sensor_index >= active_sensor_count) {
        return;
    }
    active_gyro_biases[sensor_index] = gyro_bias;
}

void read_all_sensors(Sensor* sensors, bool sensor_valid[]) {
    for (uint8_t i = 0; i < active_sensor_count; i++) {
        FusionVector accelerometer;
        FusionVector gyroscope;
        const bool ok = ism330dhcx_read_motion(
            active_imus[i].i2c_port,
            active_imus[i].device_addr,
            &accelerometer,
            &gyroscope);

        if (ok) {
            sensors[i].accelerometer = accelerometer;
            sensors[i].gyroscope = FusionVectorSubtract(gyroscope, active_gyro_biases[i]);
        }
        if (sensor_valid != NULL) {
            sensor_valid[i] = ok;
        }
        sensors[i].timestamp = time_us_64();
    }
}
 

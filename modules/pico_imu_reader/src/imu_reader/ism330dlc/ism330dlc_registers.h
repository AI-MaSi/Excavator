
#ifndef ISM330DHCL_REGISTERS_H
#define ISM330DHCL_REGISTERS_H
// ISM330DHCX Register addresses
#define WHO_AM_I 0x0F
#define CTRL3_C 0x12 // THIS IS ONLY FOR TESTING, REMOVE AFTER
#define CTRL1_XL 0x10  // Accelerometer control
#define CTRL2_G 0x11   // Gyroscope control
#define CTRL6_C 0x15   // Gyro LPF1 bandwidth (FTYPE)
#define CTRL7_G 0x16   // Gyro advanced settings (not used here)
#define CTRL8_XL 0x17  // Accel LPF2 enable / cutoff
#define STATUS_REG 0x1E
#define OUTX_L_G 0x22  // Gyroscope output registers
#define OUTX_L_XL 0x28 // Accelerometer output registers
// ISM330DHCX I2C address (SDO/SA0 pin low)
#define ISM330DHCX_ADDR_DO_LOW 0x6A
// ISM330DHCX I2C address (SDO/SA0 pin high)
#define ISM330DHCX_ADDR_DO_HIGH 0x6B

// Use 0x6B if SDO/SA0 pin is high
// Expected WHO_AM_I value for ISM330DHCX
#define ISM330DHCX_ID 0x6B
#endif

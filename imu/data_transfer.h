#ifndef DATA_TRANSFER_H
#define DATA_TRANSFER_H

#include "config.h"

// Function prototypes
uint8_t calculate_checksum(const uint8_t* data, size_t length);
void send_quaternion_data(void);
void handle_binary_commands(void);
void process_binary_command(uint8_t cmd_code, const uint8_t* data, uint8_t data_length);
bool parse_config_data(const uint8_t* data, uint8_t data_length);

#endif // DATA_TRANSFER_H

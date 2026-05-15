#include "bit_ops.h"

int16_t combine_8_bits(uint8_t low, uint8_t high) {
    return (high << 8) | low;
}
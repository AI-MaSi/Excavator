

create_full_target(i2c_helpers)

target_link_libraries(i2c_helpers
INTERFACE
pico_stdlib
hardware_i2c
ism330dlc
)
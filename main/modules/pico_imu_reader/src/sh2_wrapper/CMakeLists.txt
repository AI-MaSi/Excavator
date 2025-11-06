file(GLOB_RECURSE files "*.c")

add_library(sh2_wrapper ${files})

target_include_directories(sh2_wrapper INTERFACE .)
target_link_libraries(sh2_wrapper
PRIVATE
pico_stdlib
hardware_i2c
i2c_helpers
PUBLIC
sh2
 )
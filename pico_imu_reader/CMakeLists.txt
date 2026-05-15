# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD seeed_xiao_rp2040 CACHE STRING "Board type")


cmake_minimum_required(VERSION 3.28)

# initialize pico-sdk from submodule
# note: this must happen before project()
include(pico-sdk/pico_sdk_init.cmake)

project(ism330dlc_reader)
# Disable automatic picotool installation
# set(PICO_NO_PICOTOOL 1)

message(STATUS "PICO_SDK_PATH is: ${PICO_SDK_PATH}")

# initialize the Raspberry Pi Pico SDK
pico_sdk_init()

add_subdirectory(src)

# rest of your project
add_executable(main
    main.c
    src/imu_reader/usb_descriptors.c
)

add_executable(simple_stream
    simple_stream.c
    src/imu_reader/usb_descriptors.c
)

add_executable(i2c_scan_demo
    i2c_scan_demo.c
)

# Generate WS2812 PIO header for NeoPixel status LED
pico_generate_pio_header(main ${PICO_SDK_PATH}/src/rp2_common/pico_status_led/ws2812.pio)
pico_generate_pio_header(simple_stream ${PICO_SDK_PATH}/src/rp2_common/pico_status_led/ws2812.pio)

# Our tusb_config.h lives here (must be found before any SDK fallback)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_reader
)

target_include_directories(simple_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_reader
)

# Link TinyUSB device directly (no pico_stdio_usb)
target_link_libraries(
    main
    pico_stdlib
    pico_multicore
    hardware_i2c
    hardware_pio
    tinyusb_device
    tinyusb_board
    imu_reader
)

target_link_libraries(
    simple_stream
    pico_stdlib
    hardware_i2c
    hardware_pio
    tinyusb_device
    tinyusb_board
    imu_reader
)

target_link_libraries(
    i2c_scan_demo
    pico_stdlib
    hardware_i2c
)

# No stdio over USB — we own the CDC interface directly
pico_enable_stdio_usb(main 0)
pico_enable_stdio_uart(main 0)

pico_enable_stdio_usb(simple_stream 0)
pico_enable_stdio_uart(simple_stream 0)

pico_enable_stdio_usb(i2c_scan_demo 1)
pico_enable_stdio_uart(i2c_scan_demo 0)

# create map/bin/hex/uf2 file in addition to ELF.
pico_add_extra_outputs(main)
pico_add_extra_outputs(simple_stream)

pico_add_extra_outputs(i2c_scan_demo)

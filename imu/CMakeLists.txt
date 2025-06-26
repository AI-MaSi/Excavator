# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.1)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.1.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
cmake_minimum_required(VERSION 3.12)

# Include Pico SDK cmake
include(pico_sdk_import.cmake)

# Set project name
project(imu_processor C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Pico SDK
pico_sdk_init()

# Set the board to Seeed XIAO RP2040
set(PICO_BOARD "seeed_xiao_rp2040")

# Only define the executable once with all sources
add_executable(imu_processor
    main.c
    imu_setup.c
    imu_read.c
    kalman_processing.c
    data_transfer.c
    # Include Kalman files directly
    matrix.c
    kalman.c
    cholesky.c
)

# Make sure the headers can be found
target_include_directories(imu_processor PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

# Link with the required Pico libraries
target_link_libraries(imu_processor PRIVATE
    pico_stdlib
    pico_multicore
    hardware_i2c
    hardware_gpio
)

# Create map/bin/hex/uf2 files
pico_add_extra_outputs(imu_processor)

# Enable USB output, disable UART output
pico_enable_stdio_usb(imu_processor 1)
pico_enable_stdio_uart(imu_processor 0)
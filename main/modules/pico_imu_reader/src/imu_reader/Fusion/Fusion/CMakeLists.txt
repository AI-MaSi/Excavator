message(STATUS "FUSION has been loaded!")
add_library(Fusion 
FusionAhrs.c
FusionCompass.c
FusionOffset.c
InitFusion.c)

create_headers_target(Fusion)

target_include_directories(Fusion INTERFACE Fusion_headers)

if (TARGET imu_reader)
    target_link_libraries(imu_reader INTERFACE Fusion)
    message(STATUS "LINKED fusion to imu_reader")
endif()

target_link_libraries(Fusion
PRIVATE 
pico_stdlib
hardware_i2c
hardware_rtc
imu_reader_headers
i2c_helpers_headers
)
message(STATUS "Imu reader has been added")

create_full_target(imu_reader)

# target_link_libraries(imu_reader_headers
#     INTERFACE
#     tinyusb_device
# )

add_subdirectory(ism330dlc)
add_subdirectory(settings)
add_subdirectory(i2c)
add_subdirectory(Fusion)
add_subdirectory(utils)
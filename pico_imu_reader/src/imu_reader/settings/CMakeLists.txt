message(STATUS "imu reader settings has been loaded")

create_full_target(settings)
#target_link_libraries(imu_reader_headers INTERFACE settings_headers)
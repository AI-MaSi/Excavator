message(STATUS "ism330dlc has been loaded")

create_headers_target(ism330dlc_config)
create_headers_target(ism330dlc_registers)

create_full_target(ism330dlc)
target_link_libraries(ism330dlc INTERFACE Fusion_headers)

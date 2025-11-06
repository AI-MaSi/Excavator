message(STATUS "CMake utils have been loaded")
macro(create_headers_target NAME)
    if (NOT TARGET ${NAME}_headers)
        add_library(${NAME}_headers INTERFACE)

        target_include_directories(${NAME}_headers INTERFACE ${CMAKE_CURRENT_LIST_DIR})

        if (TARGET imu_reader AND NOT ${NAME} STREQUAL "imu_reader")
            target_link_libraries(imu_reader INTERFACE ${NAME}_headers)
        endif()
    endif()
endmacro()

macro(create_impl_target NAME)
    if (NOT TARGET ${NAME})
        if (NOT TARGET ${NAME}_headers)
            message(FATAL_ERROR "${NAME} does not have headers")
        endif()

        add_library(${NAME} INTERFACE)

        target_sources(${NAME} INTERFACE
                ${CMAKE_CURRENT_LIST_DIR}/${NAME}.c
                )
        
        target_link_libraries(${NAME} INTERFACE ${NAME}_headers)
        # Add all targets to the base imu_reader
        if (TARGET imu_reader AND NOT ${NAME} STREQUAL "imu_reader")
            message(STATUS "Linked ${NAME} to imu reader target!")
            target_link_libraries(imu_reader INTERFACE ${NAME})
        endif()
    endif()
endmacro()


macro(create_full_target TARGET)
    create_headers_target(${TARGET})
    create_impl_target(${TARGET})
endmacro()
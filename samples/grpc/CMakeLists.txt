message("samples/grpc")

set(TARGET_NAME "grpc_multi_device")

add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/grpc_multi_device.cc)

target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive tim-vx)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
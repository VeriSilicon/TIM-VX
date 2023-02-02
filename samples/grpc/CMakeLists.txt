message("samples/grpc")

set(TARGET_NAME "remote_service_demo")

add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/remote_service_demo.cc)

target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive tim-vx)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
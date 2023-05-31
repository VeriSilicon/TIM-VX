message("samples/lite_multi_device")

set(TARGET_NAME "lite_multi_device")

add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/lite_multi_device.cc)

target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive tim-vx)
target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/prebuilt-sdk/viplite/build/sdk/include)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
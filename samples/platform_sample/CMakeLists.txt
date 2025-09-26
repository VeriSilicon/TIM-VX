message("samples/platform_sample")

set(TARGET_NAME "platform_sample")

add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/platform_sample.cc)

target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive tim-vx)
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

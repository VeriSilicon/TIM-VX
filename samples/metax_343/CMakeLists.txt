message("samples/metax_343")

set(TARGET_NAME "metax_343")

aux_source_directory(. ${TARGET_NAME}_SRCS)
add_executable(${TARGET_NAME} ${${TARGET_NAME}_SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive tim-vx)
target_include_directories(${TARGET_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

add_library(synapnb INTERFACE)

target_include_directories(synapnb INTERFACE ${SYNAP_DIR}/include)
target_link_libraries(synapnb INTERFACE ${SYNAP_DIR}/lib/${PLATFORM}/libsynapnb.so)

aux_source_directory(. SRC)

add_library(synap STATIC ${SRC})

target_link_libraries(synap PRIVATE synapnb ${SYNAP_DIR}/lib/${PLATFORM}/libebg_utils.so)
target_include_directories(synap PUBLIC ${CMAKE_SOURCE_DIR}/include PRIVATE ${OVXLIB_INCLUDE_DIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/tim/lite
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim)

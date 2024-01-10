if(NOT VIVANTE_SDK_DIR)
  message(FATAL_ERROR "VIVANTE_SDK_DIR is not set")
endif()

find_library(
  OPENVX_LIB
  NAMES OpenVX
  HINTS ${VIVANTE_SDK_DIR}/lib ${VIVANTE_SDK_DIR}/lib64 ${VIVANTE_SDK_DIR}/drivers
  REQUIRED
)

add_library(viv_sdk INTERFACE)
target_link_libraries(viv_sdk INTERFACE ${OPENVX_LIB})
target_include_directories(viv_sdk INTERFACE ${VIVANTE_SDK_DIR}/include)

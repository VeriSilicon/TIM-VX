list(APPEND ${TARGET_NAME}_SRCS
  "${PROJECT_SOURCE_DIR}/src/tim/vx/platform/grpc/remote_service_client.cc"
  "${PROJECT_SOURCE_DIR}/src/tim/vx/platform/grpc/remote_service_client.h"
  "${PROJECT_SOURCE_DIR}/src/tim/vx/platform/grpc/remote.cc")

find_package(Threads REQUIRED)

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(GRPCPP_REFLECTION gRPC::grpc++_reflection)
if(CMAKE_CROSSCOMPILING)
  find_program(PROTOBUF_PROTOC protoc)
else()
  set(PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(rs_proto "${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/grpc/remote_service.proto" ABSOLUTE)
get_filename_component(rs_proto_path "${rs_proto}" PATH)

# Generated sources
set(rs_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/remote_service.pb.cc")
set(rs_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/remote_service.pb.h")
set(rs_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/remote_service.grpc.pb.cc")
set(rs_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/remote_service.grpc.pb.h")
add_custom_command(
  OUTPUT "${rs_proto_srcs}" "${rs_proto_hdrs}" "${rs_grpc_srcs}" "${rs_grpc_hdrs}"
  COMMAND ${PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${rs_proto_path}"
    --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${rs_proto}"
  DEPENDS "${rs_proto}")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

list(APPEND ${TARGET_NAME}_SRCS
  ${rs_grpc_srcs}
  ${rs_grpc_hdrs}
  ${rs_proto_srcs}
  ${rs_proto_hdrs})

list(APPEND ${TARGET_NAME}_SRCS
  "${PROJECT_SOURCE_DIR}/src/tim/vx/platform/grpc/grpc_platform_client.cc"
  "${PROJECT_SOURCE_DIR}/src/tim/vx/platform/grpc/grpc_remote.cc")

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
get_filename_component(gp_proto "${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/grpc/grpc_platform.proto" ABSOLUTE)
get_filename_component(gp_proto_path "${gp_proto}" PATH)

# Generated sources
set(gp_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.pb.cc")
set(gp_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.pb.h")
set(gp_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.grpc.pb.cc")
set(gp_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.grpc.pb.h")
add_custom_command(
  OUTPUT "${gp_proto_srcs}" "${gp_proto_hdrs}" "${gp_grpc_srcs}" "${gp_grpc_hdrs}"
  COMMAND ${PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${gp_proto_path}"
    --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${gp_proto}"
  DEPENDS "${gp_proto}")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

list(APPEND ${TARGET_NAME}_SRCS
  ${gp_grpc_srcs}
  ${gp_grpc_hdrs}
  ${gp_proto_srcs}
  ${gp_proto_hdrs})

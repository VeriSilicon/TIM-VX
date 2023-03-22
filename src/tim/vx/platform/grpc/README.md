tim-vx uses gRPC to provide a remote service, by which you can deploy your model on a remote device.
## Build and run on host
1. Build and install gRPC, see [build gRPC C++](https://github.com/grpc/grpc/blob/master/BUILDING.md)
2. Build tim-vx with gRPC
```shell
$ cd ${tim_vx_root}
$ mkdir host_build && cd host_build
$ cmake .. \
    -DTIM_VX_ENABLE_PLATFORM=ON \
    -DTIM_VX_ENABLE_GRPC=ON \
    -DTIM_VX_BUILD_EXAMPLES=ON \
    -DCMAKE_PREFIX_PATH=${grpc_host_install_path}
$ make -j4
$ make install
```
3. Start server
```shell
$ export LD_LIBRARY_PATH=${tim_vx_root}/host_build/install/lib:${tim_vx_root}/prebuilt-sdk/x86_64_linux/lib:$LD_LIBRARY_PATH
$ cd ${tim_vx_root}/host_build/install/bin
$ ./grpc_platform_server 0.0.0.0:50051
```
4. Run demo

Open a new terminal
```shell
$ export LD_LIBRARY_PATH=${tim_vx_root}/host_build/install/lib:${tim_vx_root}/prebuilt-sdk/x86_64_linux/lib:$LD_LIBRARY_PATH
$ cd ${tim_vx_root}/host_build/install/bin
$ ./grpc_multi_device 0.0.0.0:50051
```
## Build for device
1. Cross-compile gRPC, see [Cross-compile gRPC](https://github.com/grpc/grpc/blob/master/BUILDING.md#cross-compiling)

note: You should keep both two install version of gPRC: host and device.

2. Build tim-vx with host gRPC and device gRPC
```shell
$ cd ${tim_vx_root}
$ mkdir device_build && cd device_build
$ cmake .. \
    -DTIM_VX_ENABLE_PLATFORM=ON \
    -DTIM_VX_ENABLE_GRPC=ON \
    -DTIM_VX_BUILD_EXAMPLES=ON \
    -DCMAKE_PREFIX_PATH=${grpc_host_install_path} \
    -DCMAKE_TOOLCHAIN_FILE=${path_to_tool_chain_file} \
    -DEXTERNAL_VIV_SDK=${tim_vx_root}/prebuilt-sdk/x86_64_linux \
    -DProtobuf_DIR=${grpc_device_install_path}/lib/cmake/protobuf \
    -DgRPC_DIR=${grpc_device_install_path}/lib/cmake/grpc \
    -Dabsl_DIR=${grpc_device_install_path}/lib/cmake/absl
$ make -j4
$ make install
```
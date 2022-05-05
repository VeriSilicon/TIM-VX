
## 准备
在执行下面命令前，确保原始TIM-VX仓库readme正常编译安装，并使用安装路径的头文件目录更新当前  
CMakeLists.txt TIM_VX_INCLUE_PATH中的，然后在当前路径执行：
1. git submodule add -b stable ../../pybind/pybind11 extern/pybind11
2. git submodule update --init
3. python3-config --includes 获取cython的头文件路径，更新CMakeLists.txt中的CYTHON_INCLUE_PATH变量

## 编译so
1. mkdir build && cd build && cmake ..
2. make -j4
3. cp libtimvx.so ../pytim/lib/timvx.so

## 测试
替换下面的TIMVX_PATH为本地真实路径  
1. export VIVANTE_SDK_DIR=TIMVX_PATH/prebuilt-sdk/x86_64_linux
2. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TIMVX_PATH/prebuilt-sdk/x86_64_linux/lib
3. python lenet_test.py
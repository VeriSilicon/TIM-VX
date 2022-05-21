
## 准备
在执行下面命令前，确保当前的已经按照localpath/TIM-VX-python/readme.md正常编译安装到本地路径下，
比如：localpath/TIM-VX-python/host_install/
1. 切换到pytim路径, cd localpath/TIM-VX-python/src/pytim
2. 使用localpath/TIM-VX-python/host_install/include 更新CMakeLists.txt 中的TIM_VX_INCLUE_PATH
3. 从https://github.com/pybind/pybind11/releases 下载 Version 2.9.2 zip包，并解压到当前路径
4. 使用localpath/TIM-VX-python/src/pytim/pybind11-2.9.2/include 更新CMakeLists.txt 中的PYBIND11_INCLUDE_PATH
5. 执行python3-config --includes 获取cython的头文件路径，更新CMakeLists.txt中的CYTHON_INCLUE_PATH

## 编译so
1. mkdir build && cd build && cmake ..
2. make -j4
3. cp libtimvx.so ../pytim/lib/timvx.so

## 测试
替换下面的localpath为本地真实路径  
1. export TIMVX_CODE_PATH=localpath/TIM-VX-python
2. export VIVANTE_SDK_DIR=$TIMVX_CODE_PATH/prebuilt-sdk/x86_64_linux
3. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIMVX_CODE_PATH/prebuilt-sdk/x86_64_linux/lib
4. 查看详细信息需要新增环境变量，export VSI_NN_LOG_LEVEL=5
5. python examples/api_test/lenet.py
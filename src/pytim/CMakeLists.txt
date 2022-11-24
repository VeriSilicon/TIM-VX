cmake_minimum_required(VERSION 3.4.1)
project(timvx)

set(CYTHON_INCLUDE_PATH "xxxxxxxxx")
set(TIM_VX_INCLUDE_PATH "xxxxxxxxx")
set(TIM_VX_LIB_PATH "xxxxxxxxx")
set(PYBIND11_INCLUDE_PATH "xxxxxxxxx")
option(BUILD_SHARED_LIBS "build shared or static lib" ON)
file(GLOB_RECURSE SRC  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

link_directories(${TIM_VX_LIB_PATH})
add_library(timvx SHARED ${SRC})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector -std=gnu99 -O2 -flax-vector-conversions -fvisibility=hidden")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector -std=c++11 -O2 -fvisibility=hidden -fvisibility-inlines-hidden")

target_include_directories(timvx PUBLIC 
    ${CYTHON_INCLUDE_PATH}
    ${TIM_VX_INCLUDE_PATH}
    ${PYBIND11_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/timvx_ops
)
target_link_libraries(timvx PUBLIC 
    tim-vx
)


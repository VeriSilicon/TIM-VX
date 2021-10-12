add_subdirectory("benchmark_test")
add_subdirectory("lenet")

if(NOT ANDROID_TOOLCHAIN)
    add_subdirectory("multi_thread_test")
endif()

if(TIM_VX_ENABLE_NBG_PARSER)
    add_subdirectory("nbg_runner")
endif()

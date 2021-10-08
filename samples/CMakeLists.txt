add_subdirectory("benchmark_test")
add_subdirectory("lenet")
add_subdirectory("nbg_runner")

if(NOT ANDROID_TOOLCHAIN)
    add_subdirectory("multi_thread_test")
endif()

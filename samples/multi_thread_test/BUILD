cc_test(
    name = "multi_thread_test",
    copts = [
        "-Werror", "-std=c++14",
    ],
    linkopts = [
        "-lpthread"
    ],
    srcs = [
        "multi_thread_test.cc",
        "lenet_asymu8_weights.h"
    ],
    deps = [
        "//:tim-vx_interface"
    ],
)

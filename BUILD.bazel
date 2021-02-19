package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

cc_library(
    name = "tim-vx_interface",
    copts = ["-std=c++14", "-Werror", "-fvisibility=default"],
    includes = [
        "include",
        "src/tim/vx",
    ],
    hdrs = [
        "include/tim/vx/context.h",
        "include/tim/vx/graph.h",
        "include/tim/vx/operation.h",
        "include/tim/vx/tensor.h",
        "include/tim/vx/types.h",
    ] + glob([
        "include/tim/vx/ops/*.h"
    ]),
    srcs = [
        "src/tim/vx/context_private.h",
        "src/tim/vx/context.cc",
        "src/tim/vx/graph_private.h",
        "src/tim/vx/graph.cc",
        "src/tim/vx/operation.cc",
        "src/tim/vx/operation_private.h",
        "src/tim/vx/tensor.cc",
        "src/tim/vx/tensor_private.h",
        "src/tim/vx/type_utils.h",
        "src/tim/vx/type_utils.cc",
    ] + glob([
        "src/tim/vx/ops/*.cc"
    ]),
    deps = [
        "//src/tim/vx/internal:ovxlibimpl",
    ],
    linkstatic = True,
    strip_include_prefix = "include",
)

cc_binary(
    name = "libtim-vx.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        "tim-vx_interface",
    ],
)

##############################################################################
# unit test
##############################################################################
cc_test (
    name = "unit_test",
    copts = ["-std=c++14", "-Werror"],
    srcs = glob(["src/tim/vx/*_test.cc"]),
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
        ":tim-vx_interface",
    ]
)

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
        "include/tim/transform/layout_inference.h"
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
        "src/tim/transform/layout_inference.cc",
        "src/tim/transform/permute_vector.h",
        "src/tim/transform/layout_infer_context.h",
    ] + glob([
        "src/tim/vx/ops/*.cc",
        "src/tim/vx/ops/*.h"
        ], exclude = ["src/tim/vx/ops/*_test.cc"]
    ) + glob(["src/tim/transform/ops/*.*"]),
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

cc_library(
    name = "tim-lite_interface",
    copts = ["-std=c++14", "-Werror", "-fvisibility=default"],
    includes = [
        "include",
        "src/tim/lite",
    ],
    hdrs = [
        "include/tim/lite/execution.h",
        "include/tim/lite/handle.h",
    ],
    srcs = [
        "src/tim/lite/execution_private.h",
        "src/tim/lite/execution.cc",
        "src/tim/lite/handle_private.h",
        "src/tim/lite/handle.cc",
    ],
    deps = [
        "//prebuilt-sdk:VIP_LITE_LIB",
    ],
    linkstatic = True,
    strip_include_prefix = "include",
)

cc_binary(
    name = "libtim-lite.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        "tim-lite_interface",
    ],
)


##############################################################################
# unit test
##############################################################################
cc_test (
    name = "unit_test",
    copts = ["-std=c++14", "-Werror"],
    srcs = [
        "src/tim/vx/test_utils.h",
    ] + glob(["src/tim/**/*_test.cc"]),
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
        ":tim-vx_interface",
    ]
)

package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

cc_library(
    name = "nbg_parser",
    includes = [
        "include"
    ],
    hdrs =
        glob(["include/tim/utils/nbg_parser/*.h"])
    ,
    srcs =
        glob(["src/tim/utils/nbg_parser/*.c"])
    ,
    linkstatic = True,
    strip_include_prefix = "include"
)

genrule(
    name = "gen_vsi_feat_ops_def",
    srcs = ["//src/tim/vx/internal:include/interface/ops.def"],
    outs = ["vsi_feat_ops_def.h"],
    cmd = "./$(location gen_vsi_feat_ops_def.sh) $(SRCS) > \"$(OUTS)\"",
    tools = ["gen_vsi_feat_ops_def.sh"]
)

cc_library(
    name = "vsi_feat_ops_def",
    hdrs = [":gen_vsi_feat_ops_def"]
)

cc_library(
    name = "tim-vx_interface",
    defines = ["BUILD_WITH_BAZEL"],
    copts = ["-std=c++14", "-Werror", "-fvisibility=default", "-pthread"],
    includes = [
        "include",
        "src/tim/vx",
        "src/tim/transform",
    ],
    hdrs = [
        "include/tim/vx/context.h",
        "include/tim/vx/builtin_op.h",
        "include/tim/vx/graph.h",
        "include/tim/vx/operation.h",
        "include/tim/vx/ops.h",
        "include/tim/vx/tensor.h",
        "include/tim/vx/types.h",
        "include/tim/vx/compile_option.h",
        "include/tim/transform/layout_inference.h",
    ] + glob([
        "include/tim/vx/ops/*.h"
    ]),
    srcs = [
        "src/tim/vx/context_private.h",
        "src/tim/vx/context.cc",
        "src/tim/vx/compile_option.cc",
        "src/tim/vx/graph_private.h",
        "src/tim/vx/graph.cc",
        "src/tim/vx/builtin_op_impl.cc",
        "src/tim/vx/builtin_op.cc",
        "src/tim/vx/builtin_op_impl.h",
        "src/tim/vx/op_impl.cc",
        "src/tim/vx/op_impl.h",
        "src/tim/vx/operation.cc",
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
        ":vsi_feat_ops_def",
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
    includes = ["third_party/half"],
    srcs = [
        "src/tim/vx/test_utils.h",
        "third_party/half/half.hpp"
    ] + glob(["src/tim/**/*_test.cc"]),
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
        ":tim-vx_interface",
    ]
)

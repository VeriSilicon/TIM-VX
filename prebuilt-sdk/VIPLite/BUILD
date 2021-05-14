package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libs",
    srcs = glob([
        "drivers/*.so",
    ]),
)

cc_library(
    name = "VIP_LITE_LIB",
    hdrs = glob([
        "include/*.h"
    ]),
    srcs = select({
        "//conditions:default": [":libs"],
    }),
    strip_include_prefix = "include",
)


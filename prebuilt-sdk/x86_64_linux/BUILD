package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libs",
    srcs = glob([
        "lib/*.so",
        "lib/*.so.*",
    ]),
)

cc_library(
    name = "VIV_SDK_LIB",
    hdrs = glob([
        "include/**/*.h"
    ]),
    srcs = select({
        "//conditions:default": [":libs"],
    }),
    strip_include_prefix = "include",
)


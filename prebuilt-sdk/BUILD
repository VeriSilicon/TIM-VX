package(default_visibility = ["//visibility:public"])

config_setting(
    name = "x86_64_linux",
    values = {"define": "target_device=x86_64_linux"},
)

config_setting(
    name = "aarch64_A311D",
    values = {"define": "target_device=aarch64_A311D"},
)

cc_library(
    name = "VIV_SDK_LIB",
    deps = select({
        "aarch64_A311D": ["@aarch64_A311D//:VIV_SDK_LIB"],
        "//conditions:default": ["//prebuilt-sdk/x86_64_linux:VIV_SDK_LIB"],
    }),
)

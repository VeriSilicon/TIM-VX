package(default_visibility = ["//visibility:public"])

config_setting(
    name = "x86_64_linux",
    values = {"define": "target_device=x86_64_linux"},
)

cc_library(
    name = "VIV_SDK_LIB",
    deps = select({
        "//conditions:default": ["//prebuilt-sdk/x86_64_linux:VIV_SDK_LIB"],
    }),
)

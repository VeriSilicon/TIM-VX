package(default_visibility = ["//visibility:public"])

config_setting(
    name = "x86_64_linux",
    values = {"define": "target_device=x86_64_linux"},
)

config_setting(
    name = "aarch64_A311D",
    values = {"define": "target_device=aarch64_A311D"},
)

config_setting(
    name = "aarch64_S905D3",
    values = {"define": "target_device=aarch64_S905D3"},
)

config_setting(
    name = "external_viv_sdk",
    values = {"define": "use_external_viv_sdk=true"},
)

cc_library(
    name = "VIV_SDK_LIB",
    deps = select({
        "external_viv_sdk": ["@external_viv_sdk//:VIV_SDK_LIB"],
        "aarch64_A311D": ["@aarch64_A311D//:VIV_SDK_LIB"],
        "aarch64_S905D3": ["@aarch64_S905D3//:VIV_SDK_LIB"],
        "//conditions:default": ["//prebuilt-sdk/x86_64_linux:VIV_SDK_LIB"],
    }),
)
config_setting(
    name = "VIPLite_aarch64_A311D",
    values = {"define": "target_device=VIPLite_aarch64_A311D"},
)

cc_library(
    name = "VIP_LITE_LIB",
    deps = select({
        "VIPLite_aarch64_A311D": ["@VIPLite_aarch64_A311D//:VIP_LITE_LIB"],
    }),
)

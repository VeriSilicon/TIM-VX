workspace(name = "TIM_VX")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

##############################################################################
# Toolchains
##############################################################################
http_archive(
    name = "aarch64_A311D",
    build_file = "@//prebuilt-sdk/x86_64_linux:BUILD",
    sha256 = "9c3fe033f6d012010c92ed1f173b5410019ec144ddf68cbc49eaada2b4737e7f",
    strip_prefix = "aarch64_A311D_D312513_A294074_R311680_T312233_O312045",
    urls = [
        "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz",
    ],
)

local_repository(
    name = 'TOOLCHAINS',
    path = 'toolchains',
)

load("@TOOLCHAINS//:toolchains.bzl", "init_toolchains")
init_toolchains()

##############################################################################
#Third party repositories
##############################################################################
http_archive(
    name = "gtest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
    ],
)


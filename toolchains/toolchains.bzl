load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def init_toolchains(name='TOOLCHAINS'):
    http_archive(
        name = "gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu",
        build_file = "@TOOLCHAINS//gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu:toolchain.BUILD",
        sha256 = "73eed74e593e2267504efbcf3678918bb22409ab7afa3dc7c135d2c6790c2345",
        strip_prefix = "gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu",
        urls = [
            "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gcc-linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz",
        ],
    )

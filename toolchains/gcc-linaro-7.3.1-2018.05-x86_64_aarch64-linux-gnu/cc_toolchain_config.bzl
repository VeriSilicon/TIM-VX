#
# Vivante Cross Toolchain configuration
#
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "tool_path",
    "feature",
    "with_feature_set",
    "flag_group",
    "flag_set")
load("//:cc_toolchain_base.bzl",
    "build_cc_toolchain_config",
    "all_compile_actions",
    "all_cpp_compile_actions",
    "all_link_actions")

tool_paths = [
    tool_path(name = "ar", path = "bin/wrapper-ar",),
    tool_path(name = "compat-ld", path = "bin/wrapper-ld",),
    tool_path(name = "cpp", path = "bin/wrapper-cpp",),
    tool_path(name = "dwp", path = "bin/wrapper-dwp",),
    tool_path(name = "gcc", path = "bin/wrapper-gcc",),
    tool_path(name = "gcov", path = "bin/wrapper-gcov",),
    tool_path(name = "ld",  path = "bin/wrapper-ld",),
    tool_path(name = "nm", path = "bin/wrapper-nm",),
    tool_path(name = "objcopy", path = "bin/wrapper-objcopy",),
    tool_path(name = "objdump", path = "bin/wrapper-objdump",),
    tool_path(name = "strip", path = "bin/wrapper-strip",),
]

def _impl(ctx):
    builtin_sysroot = "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc"

    compile_flags_feature = feature(
        name = "compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-idirafter", "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include",
                            "-idirafter", "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/7.3.1/include",
                            "-idirafter", "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/7.3.1/include-fixed",
                            "-idirafter", "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/7.3.1/install-tools/include",
                            "-idirafter", "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/7.3.1/plugin/include",
                        ],
                    ),
                    flag_group(
                        flags = [
                            "-D__arm64",
                            "-Wall", # All warnings are enabled.
                            "-Wunused-but-set-parameter", # Enable a few more warnings that aren't part of -Wall.
                            "-Wno-free-nonheap-object", # Disable some that are problematic, has false positives
                            "-fno-omit-frame-pointer", # Keep stack frames for debugging, even in opt mode.
                            "-no-canonical-prefixes",
                            "-fstack-protector",
                            "-fPIE",
                            "-fPIC",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = all_cpp_compile_actions + [ACTION_NAMES.lto_backend],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-isystem" , "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/include/c++/7.3.1",
                            "-isystem" , "external/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/include/c++/7.3.1/aarch64-linux-gnu",
                        ]
                    ),
                ],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-g",
                        ],
                    ),
                ],
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-g0",
                            "-O2",
                            "-DNDEBUG",
                            "-ffunction-sections",
                            "-fdata-sections",
                        ],
                    ),
                ],
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-lstdc++",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wl,--gc-sections",
                        ],
                    ),
                ],
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-pie",
                        ],
                    ),
                ],
            ),
        ],
    )

    cxx_builtin_include_directories = [
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//aarch64-linux-gnu/libc/usr/include)%",
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//lib/gcc/aarch64-linux-gnu/7.3.1/include)%",
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//lib/gcc/aarch64-linux-gnu/7.3.1/include-fixed)%",
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//lib/gcc/aarch64-linux-gnu/7.3.1/install-tools/include)%",
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//lib/gcc/aarch64-linux-gnu/7.3.1/plugin/include)%",
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//aarch64-linux-gnu/include/c++/7.3.1)%",
        "%package(@gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu//aarch64-linux-gnu/include/c++/7.3.1/aarch64-linux-gnu)%",
    ]

    objcopy_embed_flags_feature = feature(
        name = "objcopy_embed_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = ["objcopy_embed_data"],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-I",
                            "binary",
                        ],
                    ),
                ],
            ),
        ]
    )

    dbg_feature = feature(name = "dbg")
    opt_feature = feature(name = "opt")

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = ctx.attr.toolchain_name,
        host_system_name = "",
        target_system_name = "linux",
        target_cpu = ctx.attr.target_cpu,
        target_libc = ctx.attr.target_cpu,
        compiler = ctx.attr.compiler,
        abi_version = ctx.attr.compiler,
        abi_libc_version = ctx.attr.compiler,
        tool_paths = tool_paths,
        features = [compile_flags_feature, objcopy_embed_flags_feature, dbg_feature, opt_feature],
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        builtin_sysroot = builtin_sysroot,
    )

# DON'T MODIFY
cc_toolchain_config = build_cc_toolchain_config(_impl)

# EOF

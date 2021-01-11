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

all_compile_actions = [
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
    ACTION_NAMES.clif_match,
]

all_cpp_compile_actions = [
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def build_cc_toolchain_config(impl):
    return rule(
        implementation = impl,
        attrs = {
            "toolchain_name": attr.string(mandatory=True),
            "target_cpu": attr.string(mandatory=True),
            "compiler": attr.string(mandatory=True),
        },
        provides = [CcToolchainConfigInfo],
    )

# construct tool path for local wrappers
def toolchain_tool_path(toolchain_name, tool_name):
    return "//{toolchain_name}/bin:{tool_name}".format(toolchain_name=toolchain_name, tool_name=tool_name)

# construct package path for remote archive tools
def toolchain_package_path(toolchain_name, target):
    return "@{toolchain_name}//:{target}".format(toolchain_name=toolchain_name, target=target)

def register_toolchain(toolchain_name, target_cpu, compiler, cc_toolchain_config):
    native.filegroup(
        name = "all_files",
        srcs = [
            toolchain_tool_path(toolchain_name, "tool-wrappers"),
            toolchain_package_path(toolchain_name, "compiler_pieces"),
        ],
    )

    native.filegroup(
        name = "linker_files",
        srcs = [
            toolchain_tool_path(toolchain_name, "ar"),
            toolchain_tool_path(toolchain_name, "gcc"),
            toolchain_tool_path(toolchain_name, "ld"),
            toolchain_package_path(toolchain_name, "compiler_pieces"),
        ],
    )

    native.filegroup(
        name = "compiler_files",
        srcs = [
            toolchain_tool_path(toolchain_name, "as"),
            toolchain_tool_path(toolchain_name, "gcc"),
            toolchain_tool_path(toolchain_name, "ld"),
        ],
    )

    native.filegroup(
        name = "empty",
        srcs = [],
    )

    cc_toolchain_config(
        name = toolchain_name + "_config",
        toolchain_name = toolchain_name,
        target_cpu = target_cpu,
        compiler = compiler)

    native.cc_toolchain(
        name = "cc-compiler",
        toolchain_identifier = toolchain_name,
        toolchain_config = toolchain_name + "_config",

        all_files = ":all_files",
        compiler_files = ":compiler_files",
        dwp_files = ":empty",
        linker_files = ":linker_files",
        objcopy_files = toolchain_tool_path(toolchain_name, "objcopy"),
        strip_files = toolchain_tool_path(toolchain_name, "strip"),
        supports_param_files = 1,
    )

    native.cc_toolchain_suite(
        name = "toolchain",
        toolchains = {
            # target_cpu | compiler
            "{target_cpu}|{compiler}".format(target_cpu=target_cpu, compiler=compiler) : "cc-compiler",
        },
        visibility = ["//visibility:public"],
    )

#
# Build Vivante chipinfo for android.
#
LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

ifeq ($(AQROOT),)
$(error Please set AQROOT env first)
endif

include $(AQROOT)/Android.mk.def

ifeq ($(PLATFORM_VENDOR),1)
LOCAL_VENDOR_MODULE  := true
endif

LOCAL_SRC_FILES :=     \
            vsi_nn_context.c \
            vsi_nn_client_op.c \
            vsi_nn_graph.c  \
            vsi_nn_node_attr_template.c  \
            vsi_nn_node.c  \
            vsi_nn_ops.c  \
            vsi_nn_daemon.c \
            vsi_nn_tensor.c \
            vsi_nn_version.c \
            vsi_nn_rnn.c \
            vsi_nn_rnn_helper.c \
            vsi_nn_internal_node.c \
            vsi_nn_log.c \
            vsi_nn_graph_optimization.c \
            vsi_nn_pre_post_process.c


LOCAL_SRC_FILES +=      \
             utils/vsi_nn_code_generator.c   \
             utils/vsi_nn_binary_tree.c   \
             utils/vsi_nn_map.c   \
             utils/vsi_nn_hashmap.c   \
             utils/vsi_nn_link_list.c   \
             utils/vsi_nn_math.c   \
             utils/vsi_nn_dtype.c   \
             utils/vsi_nn_dtype_util.c   \
             utils/vsi_nn_shape_util.c   \
             utils/vsi_nn_limits.c   \
             utils/vsi_nn_tensor_op.c   \
             utils/vsi_nn_util.c \
             utils/vsi_nn_dlfcn.c \
             utils/vsi_nn_constraint_check.c


LOCAL_SRC_FILES +=      \
             quantization/vsi_nn_dynamic_fixed_point.c   \
             quantization/vsi_nn_asymmetric_affine.c   \
             quantization/vsi_nn_perchannel_symmetric_affine.c   \


LOCAL_SRC_FILES +=      \
            post/vsi_nn_post_fasterrcnn.c   \
            post/vsi_nn_post_cmupose.c

LOCAL_SRC_FILES +=      \
            cpu_backend/vsi_nn_cpu_backend.c   \
            cpu_backend/vsi_nn_cpu_backend_conv2d.c   \
            cpu_backend/vsi_nn_cpu_backend_deconv2d.c   \
            cpu_backend/npuref_interface.c


LOCAL_SRC_FILES += libnnext/vsi_nn_libnnext_resource.c \
                   libnnext/vsi_nn_vxkernel.c

LOCAL_SRC_FILES += kernel/vsi_nn_kernel.c \
                   kernel/vsi_nn_kernel_util.c \
                   kernel/vsi_nn_kernel_backend.c \
                   kernel/vsi_nn_kernel_eltwise.c \
                   kernel/vsi_nn_kernel_selector.c \
                   kernel/vsi_nn_kernel_node.c \
                   kernel/vsi_nn_kernel_param.c \
                   kernel/vsi_nn_kernel_gpu_shape_optimize.c \
                   kernel/vsi_nn_kernel_lut.c \
                   kernel/vsi_nn_spinst.c \
                   kernel/vsi_nn_sp_unit_operation.c \
                   kernel/vsi_nn_sp_lut.c \
                   kernel/vsi_nn_gpu.c

LIBNNEXT_KERNEL_SOURCES := $(wildcard $(LOCAL_PATH)/libnnext/ops/kernel/*.c)
LOCAL_SRC_FILES += $(LIBNNEXT_KERNEL_SOURCES:$(LOCAL_PATH)/%=%)

KERNEL_SOURCES := $(wildcard $(LOCAL_PATH)/kernel/cl/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/kernel/cpu/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/kernel/evis/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/kernel/vx/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/kernel/sp/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/custom/ops/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/custom/ops/kernel/evis/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/custom/ops/kernel/cl/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/custom/ops/kernel/cpu/*.c)
KERNEL_SOURCES += $(wildcard $(LOCAL_PATH)/custom/ops/kernel/sp/*.c)
LOCAL_SRC_FILES += $(KERNEL_SOURCES:$(LOCAL_PATH)/%=%)

OPERATION_SOURCES := $(wildcard $(LOCAL_PATH)/ops/*.c)
LOCAL_SRC_FILES += $(OPERATION_SOURCES:$(LOCAL_PATH)/%=%)


LOCAL_SHARED_LIBRARIES := \
    liblog \
    libjpeg \
    libGAL \
    libOpenVX \
    libVSC \
    libdl

LOCAL_C_INCLUDES += \
    external/libjpeg-turbo \
    $(AQROOT)/sdk/inc/CL \
    $(AQROOT)/sdk/inc/VX \
    $(AQROOT)/sdk/inc/ \
    $(AQROOT)/sdk/inc/HAL \
    $(LOCAL_PATH)/../include \
    $(LOCAL_PATH)/../include/ops \
    $(LOCAL_PATH)/../include/utils \
    $(LOCAL_PATH)/../include/infernce \
    $(LOCAL_PATH)/../include/client \
    $(LOCAL_PATH)/../include/cpu_backend \
    $(LOCAL_PATH)/../include/libnnext \
    $(LOCAL_PATH)/../src

LOCAL_CFLAGS :=  \
    -DLINUX \
    -D'OVXLIB_API=__attribute__((visibility("default")))' \
    -DANDROID_SDK_VERSION=$(PLATFORM_SDK_VERSION)\
        -Wno-sign-compare \
        -Wno-implicit-function-declaration \
        -Wno-sometimes-uninitialized \
        -Wno-unused-parameter \
        -Wno-enum-conversion \
        -Wno-missing-field-initializers \
        -Wno-tautological-compare \
        -Wno-missing-braces

LOCAL_MODULE:= libovxlib
LOCAL_MODULE_TAGS := optional
LOCAL_PRELINK_MODULE := false
include $(BUILD_SHARED_LIBRARY)

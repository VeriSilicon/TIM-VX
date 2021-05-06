#
# Android Makefile for TIM-VX (assuming VENDOR build)
#
# Prerequesite: Requires Vivante SDK (libOpenVX etc) to be available
#               and VIVANTE_SDK_INC to be set for include paths
#
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifeq ($(VIVANTE_SDK_INC),)
$(error Please set VIVANTE_SDK_INC path pointing to VX/CL header file locations)
endif

LOCAL_VENDOR_MODULE := true

TIMVX_SOURCES := $(wildcard $(LOCAL_PATH)/src/tim/vx/*.c**)
TIMVX_SOURCES += $(wildcard $(LOCAL_PATH)/src/tim/vx/ops/*.c**)
LOCAL_SRC_FILES := $(TIMVX_SOURCES:$(LOCAL_PATH)/%=%)

INTERNAL_SRC_PATH := $(LOCAL_PATH)/src/tim/vx/internal/src
INTERNAL_SOURCES := $(wildcard $(INTERNAL_SRC_PATH)/*.c)
INTERNAL_SOURCES += $(wildcard $(INTERNAL_SRC_PATH)/*/*.c)
LOCAL_SRC_FILES += $(INTERNAL_SOURCES:$(LOCAL_PATH)/%=%)

LIBNNEXT_KERNEL_SOURCES := $(wildcard $(INTERNAL_SRC_PATH)/libnnext/ops/kernel/*.c)
LOCAL_SRC_FILES += $(LIBNNEXT_KERNEL_SOURCES:$(LOCAL_PATH)/%=%)

KERNEL_SOURCES := $(wildcard $(INTERNAL_SRC_PATH)/kernel/cl/*.c)
KERNEL_SOURCES += $(wildcard $(INTERNAL_SRC_PATH)/kernel/cpu/*.c)
KERNEL_SOURCES += $(wildcard $(INTERNAL_SRC_PATH)/kernel/evis/*.c)
KERNEL_SOURCES += $(wildcard $(INTERNAL_SRC_PATH)/kernel/vx/*.c)
KERNEL_SOURCES += $(wildcard $(INTERNAL_SRC_PATH)/custom/ops/*.c)
KERNEL_SOURCES += $(wildcard $(INTERNAL_SRC_PATH)/custom/ops/kernel/*.c)
LOCAL_SRC_FILES += $(KERNEL_SOURCES:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES := \
    $(VIVANTE_SDK_INC) \
    $(LOCAL_PATH)/include \
    $(LOCAL_PATH)/src/tim/vx \
    $(INTERNAL_SRC_PATH)/../include \
    $(INTERNAL_SRC_PATH)/../include/ops \
    $(INTERNAL_SRC_PATH)/../include/utils \
    $(INTERNAL_SRC_PATH)/../include/client \
    $(INTERNAL_SRC_PATH)/../include/libnnext

LOCAL_SHARED_LIBRARIES := liblog libGAL libOpenVX libVSC libdl
LOCAL_STATIC_LIBRARIES := libgtest

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

LOCAL_CPP_EXTENSION := .cc
LOCAL_MODULE := libtim-vx
LOCAL_MODULE_TAGS := optional
LOCAL_PRELINK_MODULE := false
ifeq ($(shell test $(PLATFORM_SDK_VERSION) -ge 26 && echo OK),OK)
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_PATH := $(TARGET_OUT_VENDOR)/$(Target)
else
LOCAL_MODULE_PATH := $(TARGET_OUT_SHARED_LIBRARIES)
endif
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_VENDOR_MODULE := true
LOCAL_CPP_EXTENSION := .cc
LOCAL_SRC_FILES := samples/lenet/lenet_asymu8.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_SHARED_LIBRARIES := libtim-vx
LOCAL_MODULE := lenet_asymu8
LOCAL_MODULE_TAGS := optional
LOCAL_MODULE_PATH := $(TARGET_ROOT_OUT_SBIN)
LOCAL_MODULE_CASS := EXECUTABLES
include $(BUILD_EXECUTABLE)

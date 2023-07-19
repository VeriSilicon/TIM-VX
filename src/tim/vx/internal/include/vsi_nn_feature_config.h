/*****Auto generated header file, Please DO NOT modify manually!*****/
#ifndef _VSI_NN_FEATURE_CONFIG_H
#define _VSI_NN_FEATURE_CONFIG_H

#define VSI_PUBLIC_TYPE
#include <VX/vx_khr_cnn.h>
#if defined(VX_KHR_COMPATIBILITY) && (0x1==VX_KHR_COMPATIBILITY)
#include <VX/vx_khr_compatible.h>
#endif
#ifndef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
#define VSI_PERCHANNEL_QUANTIZATION_SUPPORT
#endif
#if defined(VX_INVALIDATE_HANDLE_SUPPORT) && VX_INVALIDATE_HANDLE_SUPPORT
#define VSI_INVALIDATE_HANDLE_SUPPORT
#endif
#ifndef VSI_0_D_TENSOR_SUPPORT
#define VSI_0_D_TENSOR_SUPPORT
#endif
#if defined(VX_TENSORVIEW_ON_ANY_DIM) && VX_TENSORVIEW_ON_ANY_DIM
#define VSI_CONCAT_ENHANCE_SUPPORT
#endif
#define VSI_CREATE_TENSOR_FROM_VIEW_SUPPORT

#endif

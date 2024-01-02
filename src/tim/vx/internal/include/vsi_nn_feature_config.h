/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the Software),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
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
#ifndef VSI_SWAP_HANDLE_CACHE_SUPPORT
#define VSI_SWAP_HANDLE_CACHE_SUPPORT
#endif
#define VSI_EXPORT_APIS_FOR_SETUP_GRAPH 1
#if defined(VX_SET_TENSOR_MEMPOOL_TYPE_SUPPORT) && VX_SET_TENSOR_MEMPOOL_TYPE_SUPPORT
#define VSI_CREATE_TENSOR_FROM_AXISRAM_SUPPORT
#endif
#if defined(VX_13_NN_COMPATIBLITY)
#define VSI_MAP_TENSOR_PATCH_SUPPORT
#endif

#endif

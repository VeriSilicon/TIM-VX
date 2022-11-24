/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
/** @file */
#ifndef _VSI_NN_TYPES_H_
#define _VSI_NN_TYPES_H_

#include <stdint.h>
#include "vsi_nn_platform.h"
#include "vsi_nn_feature_config.h"

#if defined(__cplusplus)
extern "C"{
#endif

#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
#define VSI_INLINE_API __inline
#else
#define VSI_INLINE_API inline
#endif

#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
    #define SIZE_T_SPECIFIER "Iu"
    #define SSIZE_T_SPECIFIER "Id"
    #ifdef VSI_40BIT_VA_SUPPORT
        #define VSI_SIZE_T_SPECIFIER "Iu"
        #define VSI_SSIZE_T_SPECIFIER "Id"
    #else
        #define VSI_SIZE_T_SPECIFIER "u"
        #define VSI_SSIZE_T_SPECIFIER "d"
    #endif
#else
    #define SIZE_T_SPECIFIER "zu"
    #define SSIZE_T_SPECIFIER "zd"
    #ifdef VSI_40BIT_VA_SUPPORT
        #define VSI_SIZE_T_SPECIFIER "zu"
        #define VSI_SSIZE_T_SPECIFIER "zd"
    #else
        #define VSI_SIZE_T_SPECIFIER "u"
        #define VSI_SSIZE_T_SPECIFIER "d"
    #endif
#endif

#if (defined(_MSC_VER))
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

/** Enumuration type */
typedef int32_t  vsi_enum;
/** Status type */
typedef int32_t  vsi_status;
/** Bool type */
typedef int32_t   vsi_bool;
/** Half */
typedef uint16_t vsi_float16;
/** Truncate float16 */
typedef uint16_t vsi_bfloat16;
/** Tensor size */
#ifdef VSI_40BIT_VA_SUPPORT
typedef size_t vsi_size_t;
typedef ssize_t vsi_ssize_t;
#else
typedef uint32_t vsi_size_t;
typedef int32_t vsi_ssize_t;
#endif

#define VSI_SIZE_T

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/** Status enum */
typedef enum
{
    VSI_FAILURE = VX_FAILURE,
    VSI_SUCCESS = VX_SUCCESS,
}vsi_nn_status_e;

/** Pad enum */
typedef enum
{
    VSI_NN_PAD_AUTO,
    VSI_NN_PAD_VALID,
    VSI_NN_PAD_SAME
} vsi_nn_pad_e;

/** reduce type enum */
typedef enum
{
    VSI_NN_REDUCTION_TYPE_NONE,
    VSI_NN_REDUCTION_TYPE_ADD,
    VSI_NN_REDUCTION_TYPE_MUL
} vsi_nn_reduction_type_e;

/** Pad mode enum */
typedef enum {
    VSI_NN_PAD_MODE_CONSTANT,
    VSI_NN_PAD_MODE_REPLICATE,
    VSI_NN_PAD_MODE_SYMMETRIC,
    VSI_NN_PAD_MODE_REFLECT,
} vsi_nn_pad_mode_e;

/**
 * @deprecated  Platform enum
 * @see vsi_nn_dim_fmt_e
 */
typedef enum
{
    VSI_NN_PLATFORM_CAFFE,
    VSI_NN_PLATFORM_TENSORFLOW
} vsi_nn_platform_e;

/** Round type enum */
typedef enum
{
    VSI_NN_ROUND_CEIL,
    VSI_NN_ROUND_FLOOR
} vsi_nn_round_type_e;

/** Optimize driction */
typedef enum
{
    VSI_NN_OPTIMIZE_FORWARD,
    VSI_NN_OPTIMIZE_BACKWARD
} vsi_nn_opt_direction_e;
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
typedef enum
{
    VSI_MEMORY_TYPE_NONE = VX_MEMORY_TYPE_NONE,
    VSI_MEMORY_TYPE_HOST = VX_MEMORY_TYPE_HOST,
    VSI_MEMORY_TYPE_DMABUF = VX_MEMORY_TYPE_DMABUF,
    VSI_MEMORY_TYPE_INERNAL = VX_MEMORY_TYPE_INTERNAL,
    VSI_MEMORY_TYPE_UNCACHED = VX_MEMORY_TYPE_HOST_UNCACHED,
    VSI_MEMORY_TYPE_PHYSICAL = VX_MEMORY_TYPE_HOST_PHYSICAL,
}vsi_memory_type_e;
#endif
/** Type enum */
typedef enum
{
    VSI_NN_TYPE_NONE = VX_TYPE_INVALID,
    VSI_NN_TYPE_INT8 = VX_TYPE_INT8,
    VSI_NN_TYPE_INT16 = VX_TYPE_INT16,
    VSI_NN_TYPE_INT32 = VX_TYPE_INT32,
    VSI_NN_TYPE_INT64 = VX_TYPE_INT64,
    VSI_NN_TYPE_UINT8 = VX_TYPE_UINT8,
    VSI_NN_TYPE_UINT16 = VX_TYPE_UINT16,
    VSI_NN_TYPE_UINT32 = VX_TYPE_UINT32,
    VSI_NN_TYPE_UINT64 = VX_TYPE_UINT64,
    VSI_NN_TYPE_FLOAT16 = VX_TYPE_FLOAT16,
    VSI_NN_TYPE_FLOAT32 = VX_TYPE_FLOAT32,
    VSI_NN_TYPE_FLOAT64 = VX_TYPE_FLOAT64,
#ifdef VSI_BOOL8_SUPPORT
    VSI_NN_TYPE_BOOL8 = VX_TYPE_BOOL8,
#else
    VSI_NN_TYPE_BOOL8 = 0x011,
#endif
#ifdef VX_TENSOR_STRIDE_X_BITS_SUPPORT
    VSI_NN_TYPE_INT4 = VX_TYPE_INT4,
    VSI_NN_TYPE_UINT4 = VX_TYPE_UINT4,
#else
    VSI_NN_TYPE_INT4 = 0x012,
    VSI_NN_TYPE_UINT4 = 0x013,
#endif
#ifdef VSI_BFLOAT16_SUPPORT
    VSI_NN_TYPE_BFLOAT16 = VX_TYPE_BFLOAT16,
#else
    VSI_NN_TYPE_BFLOAT16 = 0x81A,
#endif
    VSI_NN_TYPE_VDATA = VX_TYPE_USER_STRUCT_START + 0x1,

}vsi_nn_type_e;

typedef int32_t vsi_nn_activation_e; enum
{
    VSI_NN_ACT_NONE    = 0,
    VSI_NN_ACT_RELU    = 1,
    VSI_NN_ACT_RELU1   = 2,
    VSI_NN_ACT_RELU6   = 3,
    VSI_NN_ACT_TANH    = 4,
    VSI_NN_ACT_SIGMOID = 6,

    VSI_NN_ACT_HARD_SIGMOID = 31, /* temporary use 31*/

    //Deprecated enum, reversed only for old code
    VSI_NN_LSTMUNIT_ACT_NONE    = 0,
    VSI_NN_LSTMUNIT_ACT_RELU    = 1,
    VSI_NN_LSTMUNIT_ACT_RELU6   = 3,
    VSI_NN_LSTMUNIT_ACT_TANH    = 4,
    VSI_NN_LSTMUNIT_ACT_SIGMOID = 6,

    VSI_NN_LSTMUNIT_ACT_HARD_SIGMOID = 31,

    VSI_NN_GRU_ACT_NONE    = 0,
    VSI_NN_GRU_ACT_RELU    = 1,
    VSI_NN_GRU_ACT_RELU6   = 3,
    VSI_NN_GRU_ACT_TANH    = 4,
    VSI_NN_GRU_ACT_SIGMOID = 6,

    VSI_NN_GRU_ACT_HARD_SIGMOID = 31
};

typedef enum
{
    VSI_NN_DEPTH2SPACE_DCR = 0,
    VSI_NN_DEPTH2SPACE_CRD
} vsi_nn_depth2space_mode_e;

typedef enum
{
    VSI_NN_GRAPH_PRELOAD_VIPSRAM,
    VSI_NN_GRAPH_PRELOAD_AXISRAM
} vsi_nn_graph_attr_preload_type_e;

typedef enum _vsi_nn_node_attr_preload_type_e
{
    VSI_NN_NODE_PRELOAD_NONE,
    VSI_NN_NODE_PRELOAD_VIPSRAM,
    VSI_NN_NODE_PRELOAD_AXISRAM
} vsi_nn_node_attr_preload_type_e;

typedef enum _vsi_nn_con2d_lstm_dataformat
{
    CONV2D_LSTM_CHANNELS_LAST,
    CONV2D_LSTM_CHANNELS_FIRST
} vsi_nn_con2d_lstm_dataformat;

typedef enum _vsi_nn_yuv_type
{
    VSI_NN_YUV_TYPE_YUYV422,
    VSI_NN_YUV_TYPE_UYUV422
}vsi_nn_yuv_type;

/** Deprecated */
typedef uint32_t vsi_nn_size_t;

/** Tensor id type */
typedef uint32_t vsi_nn_tensor_id_t;

/** Node id type */
typedef uint32_t vsi_nn_node_id_t;

/** @see _vsi_nn_graph */
typedef struct _vsi_nn_graph vsi_nn_graph_t;

/** @see _vsi_nn_node */
typedef struct _vsi_nn_node vsi_nn_node_t;

/** @see _vsi_nn_tensor */
typedef struct _vsi_nn_tensor vsi_nn_tensor_t;

#if defined(__cplusplus)
}
#endif

#endif

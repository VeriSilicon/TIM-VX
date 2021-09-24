/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
 typedef enum
{
    LAYERNORM_KERNEL,
    LAYERNORM_2D_KERNEL,
    SUMSQR_KERNEL,
    SUMSQR_2D_KERNEL,
    LAYERNORM_WH_KERNEL,
    LAYERNORM_WH_2D_KERNEL,
} _kernel_type_e;

#define KERNEL_SOURCE_1    "layer_normalization"
#define KERNEL_SOURCE_2    "layer_normalization_2d"
#define KERNEL_SOURCE_3    "layer_normalization_u8_f16"
#define KERNEL_SOURCE_4    "layer_normalization_wh_u8"
#define KERNEL_SOURCE_5    "layer_normalization_wh_f16"
#define KERNEL_SOURCE_6    "layer_normalization_i16"
#define KERNEL_SOURCE_7    "layer_normalization_wh_i16"
#define KERNEL_SOURCE_8    "layer_normalization_scale_f32"
#define KERNEL_SOURCE_9    "layer_normalization_scale_f32_2d"
#define KERNEL_SOURCE_10   "layer_normalization_scale_f32_bf16"

#define HASH_LAYERNORM_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_LAYERNORM_SH_KERNEL_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define HASH_LAYERNORM_SH_KERNEL_SCALE_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_"#SRC0_TYPE"F32to"#DST_TYPE)

#define HASH_LAYERNORM_SH_KERNEL_SCALE_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_"#SRC0_TYPE"F32to"#DST_TYPE"_2D")

// normalization
#define HASH_LAYERNORM_KEY(_input0_type, _input2_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_input2_type << 16) | (_output_type << 8) | _reshape_flag)

#define TENSOR_LAYERNORM_KERNELS(IN0_TYPE, SCALE_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, SCALE_TYPE, OUT_TYPE, LAYERNORM_KERNEL), \
        HASH_LAYERNORM_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_KERNELS_2D(IN0_TYPE, SCALE_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, SCALE_TYPE, OUT_TYPE, LAYERNORM_2D_KERNEL), \
        HASH_LAYERNORM_SH_KERNEL_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_SCALE_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F32, OUT_TYPE, LAYERNORM_KERNEL), \
        HASH_LAYERNORM_SH_KERNEL_SCALE_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_SCALE_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F32, OUT_TYPE, LAYERNORM_2D_KERNEL), \
        HASH_LAYERNORM_SH_KERNEL_SCALE_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

// greater than max size
#define HASH_SUMSQR_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layernorm_wh_sumSqr_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_SUMSQR_SH_KERNEL_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layernorm_wh_sumSqr_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define HASH_LAYERNORM_WH_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layernorm_wh_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_LAYERNORM_WH_SH_KERNEL_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layernorm_wh_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_SUMSQR_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F16, OUT_TYPE, SUMSQR_KERNEL), \
        HASH_SUMSQR_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SUMSQR_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F16, OUT_TYPE, SUMSQR_2D_KERNEL), \
        HASH_SUMSQR_SH_KERNEL_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_WH_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F16, OUT_TYPE, LAYERNORM_WH_KERNEL), \
        HASH_LAYERNORM_WH_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_WH_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F16, OUT_TYPE, LAYERNORM_WH_2D_KERNEL), \
        HASH_LAYERNORM_WH_SH_KERNEL_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _layernorm_kernel_map[] =
{
    // Register kernel here
    TENSOR_LAYERNORM_KERNELS( U8, F16, U8, KERNEL_SOURCE_1 )
    TENSOR_LAYERNORM_KERNELS_2D( U8, F16, U8, KERNEL_SOURCE_2 )
    TENSOR_LAYERNORM_KERNELS( U8, F16, F16, KERNEL_SOURCE_3 )
    TENSOR_LAYERNORM_KERNELS_2D( U8, F16, F16, KERNEL_SOURCE_3 )
    TENSOR_LAYERNORM_KERNELS( U8, F32, F16, KERNEL_SOURCE_3 )
    TENSOR_LAYERNORM_KERNELS_2D( U8, F32, F16, KERNEL_SOURCE_3 )

    TENSOR_LAYERNORM_KERNELS( F16, F16, F16, KERNEL_SOURCE_1 )
    TENSOR_LAYERNORM_KERNELS_2D( F16, F16, F16, KERNEL_SOURCE_2 )
    TENSOR_LAYERNORM_KERNELS( F16, F16, U8, KERNEL_SOURCE_1 )
    TENSOR_LAYERNORM_KERNELS_2D( F16, F16, U8, KERNEL_SOURCE_2 )
    TENSOR_LAYERNORM_KERNELS( I16, F16, I16, KERNEL_SOURCE_6 )
    TENSOR_LAYERNORM_KERNELS_2D( I16, F16, I16, KERNEL_SOURCE_6 )

    TENSOR_LAYERNORM_SCALE_KERNELS( U8, U8, KERNEL_SOURCE_8 )
    TENSOR_LAYERNORM_SCALE_KERNELS_2D( U8, U8, KERNEL_SOURCE_9 )
    TENSOR_LAYERNORM_SCALE_KERNELS( I8, I8, KERNEL_SOURCE_8 )
    TENSOR_LAYERNORM_SCALE_KERNELS_2D( I8, I8, KERNEL_SOURCE_9 )
    TENSOR_LAYERNORM_SCALE_KERNELS( I16, I16, KERNEL_SOURCE_8 )
    TENSOR_LAYERNORM_SCALE_KERNELS_2D( I16, I16, KERNEL_SOURCE_9 )
    TENSOR_LAYERNORM_SCALE_KERNELS( F16, F16, KERNEL_SOURCE_8 )
    TENSOR_LAYERNORM_SCALE_KERNELS_2D( F16, F16, KERNEL_SOURCE_9 )
    TENSOR_LAYERNORM_SCALE_KERNELS( BF16, BF16, KERNEL_SOURCE_10 )
    TENSOR_LAYERNORM_SCALE_KERNELS_2D( BF16, BF16, KERNEL_SOURCE_10 )
};

static const _kernel_map_type _sumsqr_kernel_map[] =
{
    // Register kernel here
    TENSOR_SUMSQR_KERNELS( U8, F32, KERNEL_SOURCE_4 )
    TENSOR_SUMSQR_KERNELS_2D( U8, F32, KERNEL_SOURCE_4 )
    TENSOR_SUMSQR_KERNELS( F16, F32, KERNEL_SOURCE_5 )
    TENSOR_SUMSQR_KERNELS_2D( F16, F32, KERNEL_SOURCE_5 )
    TENSOR_SUMSQR_KERNELS( I16, F32, KERNEL_SOURCE_7 )
    TENSOR_SUMSQR_KERNELS_2D( I16, F32, KERNEL_SOURCE_7 )

    TENSOR_LAYERNORM_WH_KERNELS( U8, U8, KERNEL_SOURCE_4 )
    TENSOR_LAYERNORM_WH_KERNELS_2D( U8, U8, KERNEL_SOURCE_4 )
    TENSOR_LAYERNORM_WH_KERNELS( U8, F16, KERNEL_SOURCE_4 )
    TENSOR_LAYERNORM_WH_KERNELS_2D( U8, F16, KERNEL_SOURCE_4 )
    TENSOR_LAYERNORM_WH_KERNELS( F16, F16, KERNEL_SOURCE_5 )
    TENSOR_LAYERNORM_WH_KERNELS_2D( F16, F16, KERNEL_SOURCE_5 )
    TENSOR_LAYERNORM_WH_KERNELS( I16, I16, KERNEL_SOURCE_7 )
    TENSOR_LAYERNORM_WH_KERNELS_2D( I16, I16, KERNEL_SOURCE_7 )
};

/*
 * Kernel params
 */

static vx_param_description_t _layernorm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _sumSqr_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _layernorm_wh_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

#define _LAYERNORM_PARAM_NUM    _cnt_of_array( _layernorm_kernel_param_def )
#define _SUMSQR_PARAM_NUM       _cnt_of_array( _sumSqr_kernel_param_def )
#define _LAYERNORM_WH_PARAM_NUM    _cnt_of_array( _layernorm_wh_kernel_param_def )

/*
 * Kernel initializer
 */

DEF_KERNEL_INITIALIZER(_layernorm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float scaleIn = 1;
    float scaleOut = 1;
    float output_zp = 0;
    int32_t input_zp = 0;
    int32_t iter = 0;
    int32_t sumInZp = 0;
    int32_t tmpZp1 = 0;
    int32_t tmpZp2 = 0;
    float e2InScale = 0;
    int32_t height = 0, width = 0, chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    input_shape  = attr[0]->shape;

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        input_zp     = attr[0]->asymm.zero_point;
        scaleIn      = attr[0]->asymm.scale;
    }
    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        if (attr[0]->dfp.fl > 0)
        {
            scaleIn = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            scaleIn = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
        input_zp = 0;
    }
    else if (attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        scaleIn = 1;
        input_zp = 0;
    }

    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        output_zp    = (float)attr[2]->asymm.zero_point;
        scaleOut     = 1.0f / attr[2]->asymm.scale;
    }
    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        if (attr[2]->dfp.fl > 0)
        {
            scaleOut = (float)((int64_t)1 << attr[2]->dfp.fl);
        }
        else
        {
            scaleOut = (1.0f / (float)((int64_t)1 << -attr[2]->dfp.fl));
        }
        output_zp = 0;
    }
    else if (attr[2]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        scaleOut = 1;
        output_zp = 0.0f;
    }

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)((input_shape->size <= 2) ? 1 : input_shape->data[2]);

    iter = ((width + 15) / 16) * 16;
    sumInZp = input_zp * iter * (-1);
    tmpZp1 = (-2) * input_zp;
    tmpZp2 = iter * input_zp * input_zp;
    e2InScale = scaleIn * scaleIn;

    shaderParam.global_scale[0]  = width;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = 1;
    shaderParam.global_size[1]   = height;
    shaderParam.global_size[2]   = chn;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        float dimRatio = 1.0f / (float)width;
        float dimRatio_scale = dimRatio * scaleIn;
        gpu_dp_inst_t uniFp16SumSqr_dp8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t UniFP16toFP32Lo4_dp4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf4_dp4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertSecFp16Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumU8_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSqrSum_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert1stUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert2ndUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert3rdUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert4thUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t UniPackFP16even_2x8 = {{
           0x11111111, // TCfg
           0x11110000, // ASelt
           0x06040200, 0x06040200, // ABin
           0x22222222, // BSelt
           0x00000000, 0x00000000, // BBin
           0x00000100, // AccumType, ConstantType, and PostShift
           0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniInt16SumSqr_dp8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        uint32_t pack_key      = 0;
#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 16) | (OUT_TYPE << 8))

        pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, attr[2]->dtype );

        status  = vsi_nn_kernel_gpu_add_param(node, "width", &width);
        status |= vsi_nn_kernel_gpu_add_param(node, "dimRatio", &dimRatio);
        CHECK_STATUS_FAIL_GOTO(status, OnError );

        switch( pack_key )
        {
            case _PACK_SELECT_KEY( U8, F16, F16 ):
            case _PACK_SELECT_KEY( U8, F32, F16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "UniPackFP16even_2x8",
                        &UniPackFP16even_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert3rdUint8SubZpToFp32_4x4",
                        &uniConvert3rdUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert4thUint8SubZpToFp32_4x4",
                        &uniConvert4thUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp2", &tmpZp2);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( U8,  F16, U8 ):
            case _PACK_SELECT_KEY( F16, F16, F16 ):
            case _PACK_SELECT_KEY( F16, F16, U8 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2",
                        &uniFp16SumSqr_dp8x2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractHalf4_dp4x4",
                        &uniExtractHalf4_dp4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                        &uniConvertInt32toUint8_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert3rdUint8SubZpToFp32_4x4",
                        &uniConvert3rdUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert4thUint8SubZpToFp32_4x4",
                        &uniConvert4thUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecFp16Fp32_4x4",
                        &uniConvertSecFp16Fp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4",
                        &UniFP16toFP32Lo4_dp4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp2", &tmpZp2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &scaleOut);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( I16, F16, I16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniInt16SumSqr_dp8x2",
                        &uniInt16SumSqr_dp8x2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                        &uniConvertInt32toUint8_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecFp16Fp32_4x4",
                        &uniConvertSecFp16Fp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4",
                        &UniFP16toFP32Lo4_dp4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &scaleOut);
                    status |= vsi_nn_kernel_gpu_add_param(node, "dimRatio_scale", &dimRatio_scale);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( U8,  F32, U8 ):
            case _PACK_SELECT_KEY( F16, F32, F16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2",
                        &uniFp16SumSqr_dp8x2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractHalf4_dp4x4",
                        &uniExtractHalf4_dp4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                        &uniConvertInt32toUint8_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert3rdUint8SubZpToFp32_4x4",
                        &uniConvert3rdUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert4thUint8SubZpToFp32_4x4",
                        &uniConvert4thUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4",
                        &UniFP16toFP32Lo4_dp4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp2", &tmpZp2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &scaleOut);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( I16, F32, I16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniInt16SumSqr_dp8x2",
                        &uniInt16SumSqr_dp8x2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                        &uniConvertInt32toUint8_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4",
                        &UniFP16toFP32Lo4_dp4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &scaleOut);
                    status |= vsi_nn_kernel_gpu_add_param(node, "dimRatio_scale", &dimRatio_scale);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( BF16, F32, BF16 ):
                {
                    status  = vsi_nn_kernel_gpu_add_param( node,
                                "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                                "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                                "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            default:
                VSI_ASSERT( FALSE );
                return VSI_FAILURE;
        }
#undef _PACK_SELECT_KEY
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_sumsqr_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float scaleIn = 1.0f;
    int32_t input_zp = 0;
    vx_uint32 iter = 0;
    int32_t sumInZp = 0;
    int32_t tmpZp1 = 0;
    float tmpZp2 = 0;
    float e2InScale = 0;
    float rowSumScale = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    input_shape  = attr[0]->shape;

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        input_zp     = attr[0]->asymm.zero_point;
        scaleIn      = attr[0]->asymm.scale;
    }
    else if (attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        if (attr[0]->dfp.fl > 0)
        {
            scaleIn = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            scaleIn = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
        input_zp = 0;
    }

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);
    iter = height * 16;

    e2InScale = scaleIn * scaleIn;
    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        sumInZp = input_zp * iter * (-1);
        tmpZp1 = (-2) * input_zp;
        tmpZp2 = input_zp * input_zp * e2InScale;
        rowSumScale = height * 16 * tmpZp2;
    }

    shaderParam.global_scale[0]  = 1;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.local_size[0]  = 16;
    shaderParam.local_size[1]  = 1;
    shaderParam.local_size[2]  = 1;

    if (attr[0]->dtype == I8 || attr[0]->dtype == U8)
    {
        shaderParam.global_size[0]   = (width + 255) / 256 * 16;
    }
    else if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        shaderParam.global_size[0]   = (width + 127) / 128 * 16;
    }
    shaderParam.global_size[1]   = chn;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    if (attr[0]->dtype == U8)
    {
        gpu_dp_inst_t uniSumU8_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSqrSum_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
        status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
        status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
        status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
        status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
        status |= vsi_nn_kernel_gpu_add_param(node, "rowSumScale", &rowSumScale);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
    else if (attr[0]->dtype == F16)
    {
        gpu_dp_inst_t uniFp16SumSqr_dp8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status = vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
    else if (attr[0]->dtype == I16)
    {
        gpu_dp_inst_t uniInt16SumSqr_dp8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniInt16SumSqr_dp8x2", &uniInt16SumSqr_dp8x2);
        status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
        status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

    status = vsi_nn_kernel_gpu_add_param(node, "width", &width);
    status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_layernorm_wh_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float scaleIn = 1.0f;
    float scaleOut = 1.0f;
    float output_zp = 0;
    int32_t input_zp = 0;
    float dimRatio = 0;
    vx_uint32 group_num = 0;
    vx_int32 height = 0, width = 0, chn = 0, height_chn_org = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[4] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    input_shape  = attr[0]->shape;

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        input_zp     = attr[0]->asymm.zero_point;
        scaleIn      = attr[0]->asymm.scale;
    }
    else if (attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        if (attr[0]->dfp.fl > 0)
        {
            scaleIn = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            scaleIn = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
        input_zp = 0;
    }

    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        output_zp    = (float)attr[2]->asymm.zero_point;
        scaleOut     = 1.0f / attr[2]->asymm.scale;
    }
    else if (attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        if (attr[2]->dfp.fl > 0)
        {
            scaleOut = (float)((int64_t)1 << attr[2]->dfp.fl);
        }
        else
        {
            scaleOut = (1.0f / (float)((int64_t)1 << -attr[2]->dfp.fl));
        }
        output_zp = 0;
    }

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);
    height_chn_org = (int32_t)((input_shape->size > 2 ? input_shape->data[2] : 1) / chn);

    dimRatio = (float)(1.0 / (width * height));

    group_num = (width + 255) / 256;
    if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        group_num = (width + 127) / 128;
    }

    shaderParam.global_scale[0]  = 8;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
                                        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = (chn + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1];
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t UniFP16toFP32Lo4_dp4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertSecFp16Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert1stUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert2ndUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertHalfToFp16_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };

        uint32_t pack_key      = 0;
#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (OUT_TYPE << 8))

        pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[2]->dtype );

        status  = vsi_nn_kernel_gpu_add_param(node, "height", &height);
        status |= vsi_nn_kernel_gpu_add_param(node, "height_depth", &height_chn_org);
        status |= vsi_nn_kernel_gpu_add_param(node, "dimRatio", &dimRatio);
        status |= vsi_nn_kernel_gpu_add_param(node, "group_num", &group_num);
        status |= vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4", &UniFP16toFP32Lo4_dp4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecFp16Fp32_4x4", &uniConvertSecFp16Fp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &scaleOut);
        CHECK_STATUS_FAIL_GOTO(status, OnError );

        switch( pack_key )
        {
            case _PACK_SELECT_KEY( U8, U8 ):
            case _PACK_SELECT_KEY( U8, F16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalfToFp16_2x8",
                        &uniConvertHalfToFp16_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( F16, F16 ):
            case _PACK_SELECT_KEY( F16, U8 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvertHalfToFp16_2x8",
                        &uniConvertHalfToFp16_2x8);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( I16, I16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert2ndUint8SubZpToFp32_4x4",
                        &uniConvert2ndUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            default:
                VSI_ASSERT( FALSE );
                return VSI_FAILURE;
        }
#undef _PACK_SELECT_KEY
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    return status;
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t reshape2D
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;
    _kernel_type_e kernel_type = LAYERNORM_KERNEL;

    if (reshape2D)
    {
        kernel_type = LAYERNORM_2D_KERNEL;
    }

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_LAYERNORM_KEY( input0_dtype, input2_dtype, output_dtype, kernel_type );

    for( i = 0; i < _cnt_of_array(_layernorm_kernel_map); i ++ )
    {
        if ( _layernorm_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_layernorm_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _layernorm_kernel_map[i].function_name );
        kernel->info.parameters = _layernorm_kernel_param_def;
        kernel->info.numParams = _LAYERNORM_PARAM_NUM;
        kernel->info.initialize = _layernorm_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _layernorm_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _layernorm_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_status _query_kernel_wh
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel_sumSqr,
    vsi_nn_kernel_t* kernel,
    _kernel_type_e is2D_sumsqr,
    _kernel_type_e is2D_wh
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_LAYERNORM_KEY( input0_dtype, input2_dtype, F32, is2D_sumsqr );

    for( i = 0; i < _cnt_of_array(_sumsqr_kernel_map); i ++ )
    {
        if ( _sumsqr_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_sumsqr_kernel_map) )
    {
        snprintf( kernel_sumSqr->info.name, VX_MAX_KERNEL_NAME, "%s",  _sumsqr_kernel_map[i].function_name );
        kernel_sumSqr->info.parameters = _sumSqr_kernel_param_def;
        kernel_sumSqr->info.numParams = _SUMSQR_PARAM_NUM;
        kernel_sumSqr->info.initialize = _sumsqr_initializer;

        vsi_nn_kernel_add_source( kernel_sumSqr, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _sumsqr_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_sumSqr, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _sumsqr_kernel_map[i].source_name );
    }

    key = HASH_LAYERNORM_KEY( input0_dtype, input2_dtype, output_dtype, is2D_wh );

    for( i = 0; i < _cnt_of_array(_sumsqr_kernel_map); i ++ )
    {
        if ( _sumsqr_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_sumsqr_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _sumsqr_kernel_map[i].function_name );
        kernel->info.parameters = _layernorm_wh_kernel_param_def;
        kernel->info.numParams = _LAYERNORM_WH_PARAM_NUM;
        kernel->info.initialize = _layernorm_wh_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _sumsqr_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _sumsqr_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel_wh() */

static vsi_nn_kernel_node_t _setup_wh
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL, rs_gamma = NULL, rs_beta = NULL;
    vsi_nn_kernel_node_param_t sumSqr_node_params[_SUMSQR_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t node_params[_LAYERNORM_WH_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t tmp_node = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    _kernel_type_e is2D_sumsqr = SUMSQR_2D_KERNEL;
    _kernel_type_e is2D_wh = LAYERNORM_WH_2D_KERNEL;
    vsi_nn_kernel_t * kernel_sumSqr = NULL;
    vsi_nn_tensor_t * tensor_sumSqr = NULL;
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" );

    int32_t axis[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t axis_num  = 1;
    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }};
    uint32_t axis_size = 0;
    uint32_t rank_in = 0, rank_para = 0;
    vsi_size_t outer_size = 1;
    uint32_t i = 0;

    for(i = 1; i < inputs[0]->attr.dim_num; i++)
    {
        outer_size *= inputs[0]->attr.size[i];
    }

    status = vsi_nn_kernel_optimize_tensor_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num,
        axis, axis_num, new_shape[0], &rank_in, new_axis, &axis_size);
    if ( status == FALSE || axis_size > 2)
    {
        return NULL;
    }

    status = vsi_nn_kernel_optimize_tensor_shape(
        inputs[1]->attr.size, inputs[1]->attr.dim_num,
        axis, axis_num, new_shape[1], &rank_para, new_axis, &axis_size);
    if ( status == FALSE || axis_size > 2)
    {
        return NULL;
    }

    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape[0], rank_in);

    rs_beta = vsi_nn_kernel_tensor_reshape(inputs[1]->t, new_shape[1], rank_para);

    rs_gamma = vsi_nn_kernel_tensor_reshape(inputs[2]->t, new_shape[1], rank_para);

    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape[0], rank_in);

    if (rank_in > 2)
    {
        is2D_sumsqr = SUMSQR_KERNEL;
        is2D_wh = LAYERNORM_WH_KERNEL;
    }

    kernel_sumSqr = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
    // Assign unique_id
    kernel_sumSqr->unique_id = kernel->unique_id;

    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = ((new_shape[0][0] + 255) / 256) * 4;

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
    {
        attr.size[0] = ((new_shape[0][0] + 127) / 128) * 4;
    }
    attr.size[1] = outer_size;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 4;
    tensor_sumSqr = vsi_nn_CreateTensor( graph, &attr );

    status = _query_kernel_wh(inputs, outputs, kernel_sumSqr, kernel, is2D_sumsqr, is2D_wh);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    {
        tmp_node = vsi_nn_kernel_create_node( graph, kernel_sumSqr );
        if (tmp_node)
        {
            sumSqr_node_params[0] = rs_input;
            sumSqr_node_params[1] = (vsi_nn_kernel_node_param_t)tensor_sumSqr->t;

            status  = vsi_nn_kernel_node_pass_param( tmp_node, sumSqr_node_params,
                        _SUMSQR_PARAM_NUM );
            CHECK_STATUS(status);
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = 0;
                border.constant_value.U16 = 0;
                if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
                {
                    border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
                }
                status = vxSetNodeAttribute( (vx_node)tmp_node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }

    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if (node)
        {
            uint32_t index = 0;
            node_params[index++] = rs_input;
            node_params[index++] = rs_beta;
            node_params[index++] = rs_gamma;
            node_params[index++] = (vsi_nn_kernel_node_param_t)tensor_sumSqr->t;
            node_params[index++] = rs_output;
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );

            status  = vsi_nn_kernel_node_pass_param( node, node_params,
                        _LAYERNORM_WH_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &node_params[5] );
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = 0;
                border.constant_value.U16 = 0;
                if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
                {
                    border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
                }
                status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }

final:
    if (rs_beta)
    {
        vsi_nn_kernel_tensor_release( &rs_beta );
    }
    if (rs_gamma)
    {
        vsi_nn_kernel_tensor_release( &rs_gamma );
    }
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    if ( kernel_sumSqr )
    {
        vsi_nn_kernel_release( &kernel_sumSqr );
    }
    if ( tensor_sumSqr )
    {
        vsi_nn_ReleaseTensor( &tensor_sumSqr );
    }
    if (tmp_node) {vsi_nn_kernel_node_release( &tmp_node );}

    return node;
}

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_LAYERNORM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL, rs_gamma = NULL, rs_beta = NULL;
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" );
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    int32_t rs_flg = 0;
    int32_t optFlg = 0;

    if (input_size[0] >= GPU_TENSOR_MAX_WIDTH)
    {
       node = _setup_wh(graph, inputs, input_num, outputs, output_num, params, kernel);
       goto final;
    }

    if ((input_size[1] * input_size[2] < GPU_TENSOR_MAX_WIDTH)
        && dims_num > 2)
    {
        rs_flg = 1;
    }
    optFlg = rs_flg || (outputs[0]->attr.dim_num < 3);

    status = _query_kernel( inputs, outputs, kernel, optFlg);
    if (VSI_SUCCESS != status)
    {
        goto final;
    }

    if (rs_flg)
    {
        vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
        shape[0] = inputs[0]->attr.size[0];
        shape[1] = inputs[0]->attr.size[1] * inputs[0]->attr.size[2];
        shape[2] = 1;
        shape[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
        rs_input = vsi_nn_kernel_tensor_reshape( inputs[0]->t, shape, 4 );

        shape[0] = outputs[0]->attr.size[0];
        shape[1] = outputs[0]->attr.size[1] * outputs[0]->attr.size[2];
        shape[2] = 1;
        shape[3] = outputs[0]->attr.dim_num > 3 ? outputs[0]->attr.size[3] : 1;
        rs_output = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shape, 4 );
    }
    if (inputs[1]->attr.dim_num < 2)
    {
        vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
        shape[0] = inputs[1]->attr.size[0];
        shape[1] = 1;
        shape[2] = 1;
        shape[3] = 1;
        rs_beta = vsi_nn_kernel_tensor_reshape( inputs[1]->t, shape, 4 );
    }
    if (inputs[2]->attr.dim_num < 2)
    {
        vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
        shape[0] = inputs[2]->attr.size[0];
        shape[1] = 1;
        shape[2] = 1;
        shape[3] = 1;
        rs_gamma = vsi_nn_kernel_tensor_reshape( inputs[2]->t, shape, 4 );
    }

    // Nomalization
    node = vsi_nn_kernel_create_node( graph, kernel );
    if (node)
    {
        uint32_t index = 0;
        if (rs_flg)
        {
            node_params[index++] = rs_input;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[0]->t;
        }
        if (inputs[1]->attr.dim_num < 2)
        {
            node_params[index++] = rs_beta;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
        }
        if (inputs[2]->attr.dim_num < 2)
        {
            node_params[index++] = rs_gamma;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
        }
        if (rs_flg)
        {
            node_params[index++] = rs_output;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)outputs[0]->t;
        }
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );

        status  = vsi_nn_kernel_node_pass_param( node, node_params,
            _LAYERNORM_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &node_params[4] );
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U8 = 0;
            border.constant_value.U16 = 0;
            if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
            {
                border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
            }
            status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }

    /* Pass parameters to node. */
final:
    if (rs_beta)
    {
        vsi_nn_kernel_tensor_release( &rs_beta );
    }
    if (rs_gamma)
    {
        vsi_nn_kernel_tensor_release( &rs_gamma );
    }
    if (rs_flg)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( layer_norm, _setup )

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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

#define KERNEL_SOURCE_1    "moments_axis0"
#define KERNEL_SOURCE_2    "moments_axis1"
#define KERNEL_SOURCE_3    "moments_axis2"
#define KERNEL_SOURCE_4    "moments_axis01"
#define KERNEL_SOURCE_5    "moments_axis012"
#define KERNEL_SOURCE_6    "moments_u8"
#define KERNEL_SOURCE_7    "moments_u8_axis012"

// Add kernel hashtable here
#define HASH_MOMENTS_KEY(_input0_type, _output_type, _axis_num, _axis0, _axis1, _axis2, _image_2d) \
    ((_input0_type<<24) | (_output_type<<20) | (_axis_num<<16) | (_axis0<<12) | (_axis1<<8) | (_axis2<<4)|(_image_2d))

#define HASH_MOMENTS_SH_KERNEL_NAME(AXIS0, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.moments_axis"#AXIS0"_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_MOMENTS_SH_KERNEL_2D_NAME(AXIS0, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.moments_axis"#AXIS0"_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define HASH_MOMENTS_TWO_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.moments_axis"#AXIS0#AXIS1"_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_MOMENTS_TWO_AXIS_SH_KERNEL_2D_NAME(AXIS0, AXIS1, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.moments_axis"#AXIS0#AXIS1"_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define HASH_MOMENTS_THREE_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, AXIS2, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.moments_axis"#AXIS0#AXIS1#AXIS2"_"#SRC0_TYPE"to"#DST_TYPE)

#define TENSOR_MOMENTS_KERNELS(IN0_TYPE, OUT_TYPE, AXIS0, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 1, AXIS0, 0, 0, 0), \
        HASH_MOMENTS_SH_KERNEL_NAME(AXIS0, IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MOMENTS_KERNELS_2D(IN0_TYPE, OUT_TYPE, AXIS0, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 1, AXIS0, 0, 0, 1), \
        HASH_MOMENTS_SH_KERNEL_2D_NAME(AXIS0, IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MOMENTS_TWO_AXIS_KERNELS(IN0_TYPE, OUT_TYPE, AXIS0, AXIS1, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 2, AXIS0, AXIS1, 0, 0), \
        HASH_MOMENTS_TWO_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(IN0_TYPE, OUT_TYPE, AXIS0, AXIS1, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 2, AXIS0, AXIS1, 0, 1), \
        HASH_MOMENTS_TWO_AXIS_SH_KERNEL_2D_NAME(AXIS0, AXIS1, IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MOMENTS_THREE_AXIS_KERNELS(IN0_TYPE, OUT_TYPE, AXIS0, AXIS1, AXIS2, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 3, AXIS0, AXIS1, AXIS2, 0), \
        HASH_MOMENTS_THREE_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, AXIS2, IN0_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } moments_map[] =
{
    TENSOR_MOMENTS_KERNELS(U8,  F16, 0,    KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(I8,  F16, 0,    KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(I16, F16, 0,    KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(F16, F16, 0,    KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(BF16,BF16,0,    KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(U8,  F16, 1,    KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(I8,  F16, 1,    KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(I16, F16, 1,    KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(F16, F16, 1,    KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(BF16,BF16,1,    KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(U8,  F16, 2,    KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(I8,  F16, 2,    KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(I16, F16, 2,    KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(F16, F16, 2,    KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(BF16,BF16,2,    KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(U8,  U8,  0,    KERNEL_SOURCE_6)
    TENSOR_MOMENTS_KERNELS(U8,  U8,  1,    KERNEL_SOURCE_6)
    TENSOR_MOMENTS_KERNELS(U8,  U8,  2,    KERNEL_SOURCE_6)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(U8,  F16, 0, 1,       KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(I8,  F16, 0, 1,       KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(I16, F16, 0, 1,       KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(F16, F16, 0, 1,       KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(BF16,BF16,0, 1,       KERNEL_SOURCE_7)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(U8,  U8,  0, 1,       KERNEL_SOURCE_6)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(U8,  F16, 0, 1, 2,  KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(I8,  F16, 0, 1, 2,  KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(I16, F16, 0, 1, 2,  KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(F16, F16, 0, 1, 2,  KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(BF16,BF16,0, 1, 2,  KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(U8,  U8,  0, 1, 2,  KERNEL_SOURCE_7)
    TENSOR_MOMENTS_KERNELS_2D(U8,  F16, 0,                KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS_2D(I8,  F16, 0,                KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS_2D(I16, F16, 0,                KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS_2D(F16, F16, 0,                KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS_2D(BF16,BF16,0,                KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS_2D(U8,  F16, 1,                KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS_2D(I8,  F16, 1,                KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS_2D(I16, F16, 1,                KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS_2D(F16, F16, 1,                KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS_2D(BF16,BF16,1,                KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS_2D(U8,  U8,  0,                KERNEL_SOURCE_6)
    TENSOR_MOMENTS_KERNELS_2D(U8,  U8,  1,                KERNEL_SOURCE_6)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(U8,  F16, 0, 1,    KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(I8,  F16, 0, 1,    KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(I16, F16, 0, 1,    KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(F16, F16, 0, 1,    KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(BF16,BF16,0, 1,    KERNEL_SOURCE_7)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS_2D(U8,  U8,  0, 1,    KERNEL_SOURCE_6)
};

/*
 * Kernel params
 */

static vx_param_description_t _moments_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MOMENTS_PARAM_NUM  _cnt_of_array( _moments_kernel_param_def )

static int32_t set_constant_border
    (
    vsi_nn_kernel_node_t node,
    int32_t value
    )
{
    vsi_status status = VSI_FAILURE;
    vx_border_t border;
    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.S32 = value;
    border.constant_value.U32 = (vx_uint32)value;
    border.constant_value.S16 = (vx_int16)value;
    border.constant_value.U8 = (vx_uint8)value;
    status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
    return status;
}

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_moments_initializer)
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

    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float     scaleIn  = 0;
    int32_t   input_zp = 0;
    vx_uint32 iter     = 0;
    int32_t   sumInZp  = 0;
    int32_t   tmpZp1   = 0;
    float     tmpZp2   = 0;
    float     e2InScale = 0;
    float     rowSumScale = 0;
    int32_t   axis     = 0;
    int32_t   axis_num = 0;
    int32_t   width    = 0;
    int32_t   height   = 0;
    int32_t   chn      = 0;
    float     dimRatio = 1.0;
    int32_t   iterSize = 16;
    float     zpScaleSqr_i16 = 0.0f;
    float     zpScale2_i16   = 0.0f;
    float     sumScale_i16   = 0.0f;
    float     output_ZP[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    float     outputScale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float     output_ZP0     = 0.0f;
    float     outputScale0   = 1;
    float     output_ZP1     = 0.0f;
    float     outputScale1   = 1.0f;

    uint32_t pack_key = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &axis);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &axis_num);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_shape  = attr[0]->shape;

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        input_zp = attr[0]->asymm.zero_point;
        scaleIn  = attr[0]->asymm.scale;
    }
    else if(attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
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
    else if(attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        input_zp = 0;
        scaleIn = 1;
    }

    if (attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        output_ZP0     = (float)attr[1]->asymm.zero_point;
        outputScale0   = 1.0f / attr[1]->asymm.scale;
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[1]->dfp.fl > 0)
        {
            outputScale0 = (float)((int64_t)1 << attr[1]->dfp.fl);
        }
        else
        {
            outputScale0 = (1.0f / (float)((int64_t)1 << -attr[1]->dfp.fl));
        }
        output_ZP0 = 0.0f;
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        outputScale0 = 1.0f;
        output_ZP0 = 0.0f;
    }

    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        output_ZP1     = (float)attr[2]->asymm.zero_point;
        outputScale1   = 1.0f / attr[2]->asymm.scale;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            outputScale1 = (float)((int64_t)1 << attr[2]->dfp.fl);
        }
        else
        {
            outputScale1 = (1.0f / (float)((int64_t)1 << -attr[2]->dfp.fl));
        }
        output_ZP1 = 0.0f;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        outputScale1 = 1.0f;
        output_ZP1 = 0.0f;
    }

    output_ZP[0] = output_ZP0;
    output_ZP[1] = output_ZP1;
    outputScale[0] = outputScale0;
    outputScale[1] = outputScale1;

    if(attr[0]->dtype == I16)
    {
        iterSize = 8;
    }

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->size > 1 ? input_shape->data[1] : 1);
    chn = (int32_t)(input_shape->size > 2 ? input_shape->data[2] : 1);

    shaderParam.global_scale[0]  = 1;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;

    if(axis_num == 1 && axis == 0)
    {
        iter = width;
        dimRatio = (float)(1.0 / (width));

        shaderParam.global_size[0]   = height;
        shaderParam.global_size[1]   = chn;
        shaderParam.global_size[2]   = 1;
    }
    else if(axis_num == 1 && axis == 1)
    {
        iter = height;
        dimRatio = (float)(1.0 / (height));

        shaderParam.global_scale[0]  = 4;
        shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
            / shaderParam.global_scale[0], 4);
        shaderParam.global_size[1]   = chn;
        shaderParam.global_size[2]   = 1;
    }
    else if(axis_num == 1 && axis == 2)
    {
        iter = chn;
        dimRatio = (float)(1.0 / (chn));

        shaderParam.global_scale[0]  = 4;
        shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
            / shaderParam.global_scale[0], 4);
        shaderParam.global_size[1]   = height;
        shaderParam.global_size[2]   = 1;
    }
    else if(axis_num == 2)
    {
        iter = height * iterSize;
        dimRatio = (float)(1.0 / (width * height));

        shaderParam.local_size[0]  = 16;
        shaderParam.local_size[1]  = 1;
        shaderParam.local_size[2]  = 1;
        shaderParam.global_size[0] = 16;
        shaderParam.global_size[1] = chn;
        shaderParam.global_size[2] = 1;
    }
    else if(axis_num == 3)
    {
        iter = height * iterSize;
        dimRatio = (float)(1.0 / (width * height * chn));

        shaderParam.local_size[0]  = 16;
        shaderParam.local_size[1]  = 1;
        shaderParam.local_size[2]  = 1;
        shaderParam.global_size[0] = 16;
        shaderParam.global_size[1] = 1;
        shaderParam.global_size[2] = 1;
    }

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    tmpZp1 = (-2) * input_zp;
    e2InScale = scaleIn * scaleIn;
    tmpZp2 = input_zp * input_zp * e2InScale;
    sumInZp = input_zp * iter * (-1);
    rowSumScale = iter * tmpZp2;

    zpScaleSqr_i16 = 8 * tmpZp2;
    zpScale2_i16 = tmpZp1 * e2InScale;
    sumScale_i16 = sumInZp * scaleIn;

#define _PACK_SELECT_KEY( IN0_TYPE, OUT0_TYPE, AXIS_NUM, FIRST_AXIS )    \
        (IN0_TYPE | (OUT0_TYPE << 8) | (AXIS_NUM << 16) | (FIRST_AXIS << 24))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, axis_num, axis);

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
        gpu_dp_inst_t uniFp16SumSqr_dp8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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
        gpu_dp_inst_t UniFP16toFP32Lo4_dp4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertHalftoFp16_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
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
        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
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

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8,  F16, 1, 0):
        case _PACK_SELECT_KEY( I8,  F16, 1, 0):
        case _PACK_SELECT_KEY( I16, F16, 1, 0):
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "rowSumScale", &rowSumScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpScaleSqr_i16", &zpScaleSqr_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpScale2_i16", &zpScale2_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumScale_i16", &sumScale_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, 1, 0):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, 1, 0):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8",
                        &uniConvBF16toF32_Part0_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8",
                        &uniConvBF16toF32_Part1_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                        &uniExtractOddData_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, 1, 1):
        case _PACK_SELECT_KEY( I8,  F16, 1, 1):
        case _PACK_SELECT_KEY( I16, F16, 1, 1):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, 1, 1):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4", &UniFP16toFP32Lo4_dp4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, 1, 1):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8",
                        &uniConvBF16toF32_Part0_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                        &uniExtractOddData_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, 1, 2):
        case _PACK_SELECT_KEY( I8,  F16, 1, 2):
        case _PACK_SELECT_KEY( I16, F16, 1, 2):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, 1, 2):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "UniFP16toFP32Lo4_dp4x4", &UniFP16toFP32Lo4_dp4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, 1, 2):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                        &uniExtractOddData_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, 2, 0):
        case _PACK_SELECT_KEY( I8,  F16, 2, 0):
        case _PACK_SELECT_KEY( I16, F16, 2, 0):
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "rowSumScale", &rowSumScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpScaleSqr_i16", &zpScaleSqr_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpScale2_i16", &zpScale2_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumScale_i16", &sumScale_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, 3, 0):
        case _PACK_SELECT_KEY( I8,  F16, 3, 0):
        case _PACK_SELECT_KEY( I16, F16, 3, 0):
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "rowSumScale", &rowSumScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpScaleSqr_i16", &zpScaleSqr_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpScale2_i16", &zpScale2_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumScale_i16", &sumScale_i16);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, 2, 0):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, 2, 0):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8",
                        &uniConvBF16toF32_Part1_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                        &uniExtractOddData_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, 3, 0):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, 3, 0):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8",
                        &uniConvBF16toF32_Part1_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                        &uniExtractOddData_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8, U8, 1, 0):
        case _PACK_SELECT_KEY( U8, U8, 1, 1):
        case _PACK_SELECT_KEY( U8, U8, 1, 2):
        case _PACK_SELECT_KEY( U8, U8, 2, 0):
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                        &uniConvertInt32toUint8_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvert1stUint8SubZpToFp32_4x4",
                        &uniConvert1stUint8SubZpToFp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                status |= vsi_nn_kernel_gpu_add_param(node, "inputZP", &input_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "rowSumScale", &rowSumScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP", &output_ZP);
                status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &outputScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP0", &output_ZP0);
                status |= vsi_nn_kernel_gpu_add_param(node, "outputScale0", &outputScale0);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP1", &output_ZP1);
                status |= vsi_nn_kernel_gpu_add_param(node, "outputScale1", &outputScale1);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8, U8, 3, 0):
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniSumU8_16x1", &uniSumU8_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniSqrSum_16x1", &uniSqrSum_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                        &uniConvertInt32toUint8_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "sumInZp", &sumInZp);
                status |= vsi_nn_kernel_gpu_add_param(node, "tmpZp1", &tmpZp1);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
                status |= vsi_nn_kernel_gpu_add_param(node, "e2InScale", &e2InScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "rowSumScale", &rowSumScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &chn);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP", &output_ZP);
                status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &outputScale);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            VSI_ASSERT( FALSE );
            break;
        }
        status = vsi_nn_kernel_gpu_add_param(node, "dimRatio", &dimRatio);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
#undef _PACK_SELECT_KEY

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
    const vsi_nn_kernel_param_t * params,
    int32_t* axis,
    int32_t axis_num,
    int32_t rs_flg
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if ( ( input0_dtype == I8 || input0_dtype == I16 ) &&
         ( inputs[0]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_DFP &&
           inputs[0]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_NONE ) )
    {
        return VSI_FAILURE;
    }

    key = HASH_MOMENTS_KEY( input0_dtype, output_dtype, axis_num, axis[0], axis[1], axis[2], rs_flg );

    for( i = 0; i < _cnt_of_array(moments_map); i++ )
    {
        if( moments_map[i].key == key )
        {
            break;
        }
    }

    if( i < _cnt_of_array(moments_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  moments_map[i].function_name );
        kernel->info.parameters = _moments_kernel_param_def;
        kernel->info.numParams = _MOMENTS_PARAM_NUM;
        kernel->info.initialize = _moments_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                moments_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                moments_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

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
    vsi_nn_kernel_node_param_t node_params[_MOMENTS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis_num  = 0;
    size_t axis_num_temp = 0;
    int32_t* axis = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "axis", &axis_num_temp);
    int32_t axis_first  = axis[0];
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 1, 1, 1, 1 } };
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };

    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t i = 0;
    uint32_t axis_size = 0;
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    vsi_bool ret = FALSE;
    vsi_bool image_2d = FALSE;
    vsi_bool is_continue_axis = TRUE;

    axis_num = (int32_t)axis_num_temp;

    for ( i = 1; i < axis_num; i++)
    {
        if ( axis[i] != (axis[i - 1] + 1) && axis[0] == 0)
        {
            is_continue_axis = FALSE;
            break;
        }
    }

    if (is_continue_axis == FALSE)
    {
        return NULL;
    }

    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            axis, axis_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            new_axis, &axis_size);

    if ( ret == FALSE || axis_size > 2)
    {
        return NULL;
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], rank_in );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[1], rank_out );
    reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
        outputs[1], shapes[1], rank_out );

    if( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[1]->attr.size,
        reshape_tensors[1]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (reshape_tensors[0]->attr.dim_num == 2 || reshape_tensors[0]->attr.size[2] == 1);
    axis_first = new_axis[0];

    status = _query_kernel( inputs, outputs, kernel, params, new_axis, axis_size, image_2d );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 3;
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _MOMENTS_PARAM_NUM,
                reshape_tensors, 1, &reshape_tensors[1], 2 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_first );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_size );
            status = vsi_nn_kernel_node_pass_param( node, node_params, _MOMENTS_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            status = set_constant_border(node, vsi_nn_get_tensor_zero_point(inputs[0]));
            CHECK_STATUS(status);
        }
    }

    for(i = 0; i < 3; i++)
    {
        if(reshape_tensors[i])
        {
            vsi_nn_ReleaseTensor(&reshape_tensors[i]);
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( moments, _setup )

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
#include "utils/vsi_nn_dtype_util.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

#define KERNEL_SOURCE_1    "cumsum"
#define KERNEL_SOURCE_2    "cumsum_2d"
#define KERNEL_SOURCE_3    "cumsum_bf16"
#define KERNEL_SOURCE_4    "cumsum_f16_u8"
#define KERNEL_SOURCE_5    "cumsum_ex_rev_axis0"
#define KERNEL_SOURCE_6    "cumsum_ex_rev_axis1"
#define KERNEL_SOURCE_7    "cumsum_ex_rev_axis2"

// Add kernel hashtable here
#define HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, EX_REV, _image_2d) \
    ((EX_REV << 24) | (AXIS << 20) | (IN_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

#define HASH_CUMSUM_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE, SOURCE) \
        { HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0, 0), \
        CVIVANTE_NAMESPACE("evis.cumsum_"#IN_DTYPE"to"#OUT_DTYPE"_axis"#AXIS), \
        SOURCE },

#define HASH_CUMSUM_KERNELS_2D( AXIS, IN_DTYPE, OUT_DTYPE, SOURCE) \
        { HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0, 1), \
        CVIVANTE_NAMESPACE("evis.cumsum_"#IN_DTYPE"to"#OUT_DTYPE"_axis"#AXIS"_2D"), \
        SOURCE },

#define HASH_CUMSUM_EX_REV_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE, SOURCE) \
        { HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1, 0), \
        CVIVANTE_NAMESPACE("evis.cumsum_ex_rev_"#IN_DTYPE"to"#OUT_DTYPE"_axis"#AXIS), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } cumsum_map[] =
{
    HASH_CUMSUM_KERNELS(0, U8,   U8,   KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(0, I8,   I8,   KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(0, I16,  I16,  KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(0, F16,  F16,  KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(0, BF16, BF16, KERNEL_SOURCE_3)
    HASH_CUMSUM_KERNELS(1, U8,   U8,   KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(1, I8,   I8,   KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(1, I16,  I16,  KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(1, F16,  F16,  KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(1, BF16, BF16, KERNEL_SOURCE_3)
    HASH_CUMSUM_KERNELS(2, U8,   U8,   KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(2, I8,   I8,   KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(2, I16,  I16,  KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(2, F16,  F16,  KERNEL_SOURCE_1)
    HASH_CUMSUM_KERNELS(2, BF16, BF16, KERNEL_SOURCE_3)
    HASH_CUMSUM_KERNELS_2D(0, U8,   U8,   KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(0, I8,   I8,   KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(0, I16,  I16,  KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(0, F16,  F16,  KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(0, BF16, BF16, KERNEL_SOURCE_3)
    HASH_CUMSUM_KERNELS_2D(1, U8,   U8,   KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(1, I8,   I8,   KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(1, I16,  I16,  KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(1, F16,  F16,  KERNEL_SOURCE_2)
    HASH_CUMSUM_KERNELS_2D(1, BF16, BF16, KERNEL_SOURCE_3)
    HASH_CUMSUM_KERNELS(0, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(0, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(0, F16,  I16, KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(1, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(1, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(1, F16,  I16, KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(2, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(2, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS(2, F16,  I16, KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS_2D(0, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS_2D(0, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS_2D(0, F16,  I16, KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS_2D(1, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS_2D(1, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_KERNELS_2D(1, F16,  I16, KERNEL_SOURCE_4)
    HASH_CUMSUM_EX_REV_KERNELS(0, U8,   U8,  KERNEL_SOURCE_5)
    HASH_CUMSUM_EX_REV_KERNELS(0, I8,   I8,  KERNEL_SOURCE_5)
    HASH_CUMSUM_EX_REV_KERNELS(0, I16,  I16, KERNEL_SOURCE_5)
    HASH_CUMSUM_EX_REV_KERNELS(0, F16,  F16, KERNEL_SOURCE_5)
    HASH_CUMSUM_EX_REV_KERNELS(1, U8,   U8,  KERNEL_SOURCE_6)
    HASH_CUMSUM_EX_REV_KERNELS(1, I8,   I8,  KERNEL_SOURCE_6)
    HASH_CUMSUM_EX_REV_KERNELS(1, I16,  I16, KERNEL_SOURCE_6)
    HASH_CUMSUM_EX_REV_KERNELS(1, F16,  F16, KERNEL_SOURCE_6)
    HASH_CUMSUM_EX_REV_KERNELS(2, U8,   U8,  KERNEL_SOURCE_7)
    HASH_CUMSUM_EX_REV_KERNELS(2, I8,   I8,  KERNEL_SOURCE_7)
    HASH_CUMSUM_EX_REV_KERNELS(2, I16,  I16, KERNEL_SOURCE_7)
    HASH_CUMSUM_EX_REV_KERNELS(2, F16,  F16, KERNEL_SOURCE_7)
    HASH_CUMSUM_EX_REV_KERNELS(1, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_EX_REV_KERNELS(1, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_EX_REV_KERNELS(1, F16,  I16, KERNEL_SOURCE_4)
    HASH_CUMSUM_EX_REV_KERNELS(2, F16,  U8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_EX_REV_KERNELS(2, F16,  I8,  KERNEL_SOURCE_4)
    HASH_CUMSUM_EX_REV_KERNELS(2, F16,  I16, KERNEL_SOURCE_4)
};

/*
 * Kernel params
 */
static vx_param_description_t _cumsum_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CUMSUM_PARAM_NUM  _cnt_of_array( _cumsum_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_cumsum_initializer)
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

    int32_t       axis    = 0;
    int32_t       exclusive = 0;
    int32_t       reverse = 0;
    int32_t       width   = 0;
    int32_t       height  = 0;
    int32_t       channel = 0;
    int32_t       w       = 1;
    int32_t       h       = 1;
    int32_t       c       = 1;
    uint32_t      dim     = 1;
    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    int32_t input_zp        = 0;
    float   input_scale     = 1.0f;
    float   output_zp       = 0;
    float   output_scale    = 1.0f;
    float   in_out_zp_scale = 1.0f;
    float   in_out_scale    = 1.0f;

    uint32_t pack_key = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &exclusive);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &reverse);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            input_scale = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            input_scale = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input_scale = attr[0]->asymm.scale;
        input_zp = attr[0]->asymm.zero_point;
    }

    if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[1]->dfp.fl > 0)
        {
            output_scale = (float)((int64_t)1 << attr[1]->dfp.fl);
        }
        else
        {
            output_scale = (1.0f / (float)((int64_t)1 << -attr[1]->dfp.fl));
        }
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        output_scale = 1.0f / attr[1]->asymm.scale;
        output_zp = (float)attr[1]->asymm.zero_point;
    }

    in_out_scale = input_scale * output_scale;
    in_out_zp_scale = (float)in_out_scale * input_zp * (-1);

    input_shape  = attr[0]->shape;
    dim     = (uint32_t)input_shape->size;
    width   = (int32_t)(input_shape->data[0]);
    height  = (int32_t)(input_shape->data[1]);
    channel = (int32_t)(dim > 2 ? input_shape->data[2] : 1);


    if (axis == 0)
    {
        w = 1;
        h = height;
        c = channel;
    }
    else if (axis == 1)
    {
        w = width;
        h = 1;
        c = channel;
    }
    else if (axis == 2)
    {
        w = width;
        h = height;
        c = 1;
    }

    shaderParam.global_scale[0]  = 8;
    if ((attr[0]->dtype == U8 || attr[0]->dtype == I8)
        && (axis > 0))
    {
        shaderParam.global_scale[0]  = 16;
    }
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = (w + shaderParam.global_scale[0] - 1) / shaderParam.global_scale[0];
    shaderParam.global_size[1]   = h;
    shaderParam.global_size[2]   = c;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE, AXIS, DIM)    \
        (IN0_TYPE | (OUT_TYPE << 8) | (AXIS << 16) | (DIM << 24))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, axis, dim);

    {
        uint16_t M0               = 0;
        int32_t  postShift        = 0;
        uint32_t multAndoutZP0[2] = {0};
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniAccSumVertF16toF16_2x8 = {{
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x33221100, 0x77665544, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001,
            0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumVertU8toI32A_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumVertU8toI32B_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00150004, 0x00370026, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumVertU8toI32C_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00190008, 0x003b002a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumVertU8toI32D_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x001d000c, 0x003f002e, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniSumHorzF16toF16A_4x4 = {{
            0x55150501, // TCfg
            0x00000000, // ASelt
            0x00100000, 0x32100210, // ABin
            0xaa2a0a02, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00003c00, 0x3c003c00, 0x3c003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumHorzF16toF16B_4x4 = {{
            0x55150501, // TCfg
            0x00000000, // ASelt
            0x00540004, 0x76540654, // ABin
            0xaa2a0a02, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00003c00, 0x3c003c00, 0x3c003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumHorzF16toF16C_2x8 = {{
            0x55551111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x37363534, // ABin
            0xaaaa2222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumHorzF16toF16_2x8 = {{
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x73727170, 0x77767574, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniSumHorzU8toI16A_4x4 = {{
            0x55150501, // TCfg
            0x00000000, // ASelt
            0x00100000, 0x32100210, // ABin
            0xaa2a0a02, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000700, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumHorzU8toI16B_8x4 = {{
            0x05550155, 0x55551555, // TCfg
            0x00418820, 0x41882000, 0x8820000a, 0x20018a41, 0x398a4188, // BinSelect
            0x00000700, // AccumType, ConstantType, and PostShift
            0x01010101, 0x00000001, 0x01010101, 0x00000101,
            0x01010101, 0x00010101, 0x01010101, 0x01010101 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSubZpI16toI16_2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00020001, 0x00030001, 0x00040001,
            0x00050001, 0x00060001, 0x00070001, 0x00080001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumHorzI16toI32A_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00310030, 0x00330032, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAccSumHorzI16toI32B_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00350034, 0x00370036, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
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
            0x11111111,  // TCfg
            0x01010101,  // ASelt
            0x01050004, 0x03070206,  // ABin
            0x22222222,  // BSelt
            0x00000000, 0x00000000,  // BBin
            0x00000600,  // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001  // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111,  // TCfg
            0x01010101,  // ASelt
            0x05050404, 0x07070606,  // ABin
            0x22222222,  // BSelt
            0x00000000, 0x00000000,  // BBin
            0x00000600,  // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001  // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111,  // TCfg
            0x11110000,  // ASelt
            0x07050301, 0x07050301,  // ABin
            0x22222222,  // BSelt
            0x00000000, 0x00000000,  // BBin
            0x00000600,  // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001  // Constant
        }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniSetZeroF16_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniSumHorzRevF16toF16A_4x4 = {{
            0x01051555, // TCfg
            0x00000000, // ASelt
            0x05674567, 0x00070067, // ABin
            0x020a2aaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00003c00,
            0x3c003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSumHorzRevF16toF16B_4x4 = {{
            0x01051555, // TCfg
            0x00000000, // ASelt
            0x01230123, 0x00030023, // ABin
            0x020a2aaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00003c00,
            0x3c003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSumHorzRevF16toF16C_2x8 = {{
            0x11115555, // TCfg
            0x00000000, // ASelt
            0x43424140, 0x07060504, // ABin
            0x2222aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniAccSumHorzRevF16toF16_2x8 = {{
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSumHorzRevU8toI16A_4x4 = {{
            0x01051555, // TCfg
            0x00000000, // ASelt
            0x05674567, 0x00070067, // ABin
            0x020a2aaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00000001,
            0x00010001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSumHorzRevU8toI16B_8x4 = {{
            0x15555555, 0x01550555, // TCfg
            0x443214c7, 0x3214c700, 0x14c70044, 0xc7000432, 0x00003214, // BinSelect
            0x00000700, // AccumType, ConstantType, and PostShift
            0x01010101, 0x01010101, 0x01010101, 0x00010101,
            0x01010101, 0x00000101, 0x01010101, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSubZpRevI16toI16_2x8 = {{
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00080001, 0x00070001, 0x00060001, 0x00050001,
            0x00040001, 0x00030001, 0x00020001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniAccSumHorzRevI16toI32A_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniAccSumHorzRevI16toI32B_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_quantize_multiplier_16bit( (double)input_scale * output_scale, &M0, &postShift);
        multAndoutZP0[0] = (uint32_t)(M0);
        multAndoutZP0[1] = (uint32_t)((attr[1]->asymm.zero_point << postShift) - input_zp * M0);
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift );

        if ((exclusive || reverse) && axis == 0)
        {
            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzRevF16toF16A_4x4", &uniSumHorzRevF16toF16A_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzRevF16toF16B_4x4", &uniSumHorzRevF16toF16B_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzRevF16toF16C_2x8", &uniSumHorzRevF16toF16C_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumHorzRevF16toF16_2x8", &uniAccSumHorzRevF16toF16_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzRevU8toI16A_4x4", &uniSumHorzRevU8toI16A_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzRevU8toI16B_8x4", &uniSumHorzRevU8toI16B_8x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSubZpRevI16toI16_2x8", &uniSubZpRevI16toI16_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumHorzRevI16toI32A_4x4", &uniAccSumHorzRevI16toI32A_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumHorzRevI16toI32B_4x4", &uniAccSumHorzRevI16toI32B_4x4 );
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8,   U8,   2, 3):
        case _PACK_SELECT_KEY( I8,   I8,   2, 3):
        case _PACK_SELECT_KEY( I16,  I16,  2, 3):
        case _PACK_SELECT_KEY( F16,  F16,  2, 3):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "channel", &channel);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "in_out_scale", &in_out_scale);
                status |= vsi_nn_kernel_gpu_add_param(node, "in_out_zp_scale", &in_out_zp_scale);
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertF16toF16_2x8", &uniAccSumVertF16toF16_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32A_4x4", &uniAccSumVertU8toI32A_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32B_4x4", &uniAccSumVertU8toI32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32C_4x4", &uniAccSumVertU8toI32C_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32D_4x4", &uniAccSumVertU8toI32D_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSetZeroF16_2x8", &uniSetZeroF16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,   U8,   1, 3):
        case _PACK_SELECT_KEY( I8,   I8,   1, 3):
        case _PACK_SELECT_KEY( I16,  I16,  1, 3):
        case _PACK_SELECT_KEY( F16,  F16,  1, 3):
        case _PACK_SELECT_KEY( U8,   U8,   1, 2):
        case _PACK_SELECT_KEY( I8,   I8,   1, 2):
        case _PACK_SELECT_KEY( I16,  I16,  1, 2):
        case _PACK_SELECT_KEY( F16,  F16,  1, 2):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "in_out_scale", &in_out_scale);
                status |= vsi_nn_kernel_gpu_add_param(node, "in_out_zp_scale", &in_out_zp_scale);
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertF16toF16_2x8", &uniAccSumVertF16toF16_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32A_4x4", &uniAccSumVertU8toI32A_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32B_4x4", &uniAccSumVertU8toI32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32C_4x4", &uniAccSumVertU8toI32C_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumVertU8toI32D_4x4", &uniAccSumVertU8toI32D_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSetZeroF16_2x8", &uniSetZeroF16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,   U8,   0, 2):
        case _PACK_SELECT_KEY( U8,   U8,   0, 3):
        case _PACK_SELECT_KEY( I8,   I8,   0, 2):
        case _PACK_SELECT_KEY( I8,   I8,   0, 3):
        case _PACK_SELECT_KEY( I16,  I16,  0, 2):
        case _PACK_SELECT_KEY( I16,  I16,  0, 3):
        case _PACK_SELECT_KEY( F16,  F16,  0, 2):
        case _PACK_SELECT_KEY( F16,  F16,  0, 3):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_zp", &input_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                status |= vsi_nn_kernel_gpu_add_param(node, "in_out_scale", &in_out_scale);
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzF16toF16A_4x4", &uniSumHorzF16toF16A_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzF16toF16B_4x4", &uniSumHorzF16toF16B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzF16toF16C_2x8", &uniSumHorzF16toF16C_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumHorzF16toF16_2x8", &uniAccSumHorzF16toF16_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzU8toI16A_4x4", &uniSumHorzU8toI16A_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumHorzU8toI16B_8x4", &uniSumHorzU8toI16B_8x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSubZpI16toI16_2x8", &uniSubZpI16toI16_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumHorzI16toI32A_4x4", &uniAccSumHorzI16toI32A_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniAccSumHorzI16toI32B_4x4", &uniAccSumHorzI16toI32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSetZeroF16_2x8", &uniSetZeroF16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, 0, 2):
        case _PACK_SELECT_KEY( BF16, BF16, 1, 2):
        case _PACK_SELECT_KEY( BF16, BF16, 0, 3):
        case _PACK_SELECT_KEY( BF16, BF16, 1, 3):
        case _PACK_SELECT_KEY( BF16, BF16, 2, 3):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &channel);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniExtractOddData_2x8", &uniExtractOddData_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8,  0, 2):
        case _PACK_SELECT_KEY( F16, U8,  1, 2):
        case _PACK_SELECT_KEY( F16, U8,  0, 3):
        case _PACK_SELECT_KEY( F16, U8,  1, 3):
        case _PACK_SELECT_KEY( F16, U8,  2, 3):
        case _PACK_SELECT_KEY( F16, I8,  0, 2):
        case _PACK_SELECT_KEY( F16, I8,  1, 2):
        case _PACK_SELECT_KEY( F16, I8,  0, 3):
        case _PACK_SELECT_KEY( F16, I8,  1, 3):
        case _PACK_SELECT_KEY( F16, I8,  2, 3):
        case _PACK_SELECT_KEY( F16, I16, 0, 2):
        case _PACK_SELECT_KEY( F16, I16, 1, 2):
        case _PACK_SELECT_KEY( F16, I16, 0, 3):
        case _PACK_SELECT_KEY( F16, I16, 1, 3):
        case _PACK_SELECT_KEY( F16, I16, 2, 3):
            {
                status = vsi_nn_kernel_gpu_add_param(node, "width", &width);
                status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
                status |= vsi_nn_kernel_gpu_add_param(node, "channel", &channel);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniAccSumVertF16toF16_2x8", &uniAccSumVertF16toF16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSumHorzF16toF16A_4x4", &uniSumHorzF16toF16A_4x4);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSumHorzF16toF16B_4x4", &uniSumHorzF16toF16B_4x4);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSumHorzF16toF16C_2x8", &uniSumHorzF16toF16C_2x8);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniAccSumHorzF16toF16_2x8", &uniAccSumHorzF16toF16_2x8);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "multAndoutZP0", &multAndoutZP0);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniSetZeroF16_2x8", &uniSetZeroF16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
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
    int32_t axis,
    int32_t is_2d,
    int32_t is_ex_rev
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    VSI_UNREFERENCED(params);

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_CUMSUM_HASH_KEY( axis, input0_dtype, output_dtype, is_ex_rev, is_2d);

    for ( i = 0; i < _cnt_of_array(cumsum_map); i ++ )
    {
        if ( cumsum_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(cumsum_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  cumsum_map[i].function_name );
        kernel->info.parameters = _cumsum_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _cumsum_kernel_param_def );
        kernel->info.initialize = _cumsum_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                cumsum_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                cumsum_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_CUMSUM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[1][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    int32_t axis       = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t exclusive  = vsi_nn_kernel_param_get_int32( params, "exclusive" );
    int32_t reverse    = vsi_nn_kernel_param_get_int32( params, "reverse" );
    int32_t axis_new   = 0;
    int32_t is_2d      = 0;
    uint32_t rs_dim    = 2;
    uint32_t i         = 0;
    int32_t is_ex_or_rev  = exclusive || reverse;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if (axis < 0)
    {
        axis += (int32_t)inputs[0]->attr.dim_num;
    }

    vsi_nn_kernel_optimize_softmax_shape(
                inputs[0]->attr.size, inputs[0]->attr.dim_num, axis,
                shapes[0], &rs_dim, &axis_new);

    if (rs_dim > 3)
    {
        return NULL;
    }

    if (rs_dim == 2 && is_ex_or_rev == 0)
    {
        is_2d = 1;
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], (vsi_size_t)rs_dim );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[0], (vsi_size_t)rs_dim );

    status = _query_kernel( inputs, outputs, kernel, params, axis_new, is_2d, is_ex_or_rev);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 2;

            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _CUMSUM_PARAM_NUM,
                reshape_tensors, 1, &reshape_tensors[1], 1 );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_new );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &exclusive );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _CUMSUM_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                vsi_nn_Float32ToDtype(0, (uint8_t*)&border.constant_value.U32, &outputs[0]->attr.dtype);
                status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }

    for (i = 0; i < 2; i++)
    {
        vsi_safe_release_tensor(reshape_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( cumsum, _setup )

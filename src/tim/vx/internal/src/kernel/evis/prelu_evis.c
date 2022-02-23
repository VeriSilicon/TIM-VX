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
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS


#define KERNEL_SOURCE0    "prelu",
#define KERNEL_SOURCE1    "prelu_BF16",

 typedef enum
{
    _3D = 0,
    _2D,
    _2D_OPT,
} vsi_nn_shader_type_e;

#define HASH_PRELU_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define PRELU_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_PRELU_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
    CVIVANTE_NAMESPACE("evis.prelu_"#IN0_TYPE#IN1_TYPE"to"#OUT_TYPE), \
        SOURCE },

#define PRELU_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE, SH_TYPE, SOURCE) \
    { HASH_PRELU_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, SH_TYPE), \
    CVIVANTE_NAMESPACE("evis.prelu_"#IN0_TYPE#IN1_TYPE"to"#OUT_TYPE#SH_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    PRELU_KERNELS(BF16, BF16, BF16, KERNEL_SOURCE1)
    PRELU_KERNELS(BF16, F16,  BF16, KERNEL_SOURCE1)
    PRELU_KERNELS(F16,  F16,  F16,  KERNEL_SOURCE0)
    PRELU_KERNELS(F16,  F16,  I16,  KERNEL_SOURCE0)
    PRELU_KERNELS(F16,  F16,  U8,   KERNEL_SOURCE0)
    PRELU_KERNELS(F16,  F16,  I8,   KERNEL_SOURCE0)
    PRELU_KERNELS(I16,  F16,  I16,  KERNEL_SOURCE0)
    PRELU_KERNELS(I16,  F16,  F16,  KERNEL_SOURCE0)
    PRELU_KERNELS(U8,   F16,  U8,   KERNEL_SOURCE0)
    PRELU_KERNELS(U8,   F16,  F16,  KERNEL_SOURCE0)
    PRELU_KERNELS(I8,   F16,  I8,   KERNEL_SOURCE0)
    PRELU_KERNELS(I8,   F16,  F16,  KERNEL_SOURCE0)
    PRELU_KERNELS(I8,   F16,  F16,  KERNEL_SOURCE0)

    PRELU_KERNELS_2D(BF16, BF16, BF16, _2D,     KERNEL_SOURCE1)
    PRELU_KERNELS_2D(BF16, F16,  BF16, _2D,     KERNEL_SOURCE1)
    PRELU_KERNELS_2D(F16,  F16,  F16,  _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(F16,  F16,  I16,  _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(F16,  F16,  U8,   _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(F16,  F16,  I8,   _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(I16,  F16,  I16,  _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(I16,  F16,  I16,  _2D_OPT, KERNEL_SOURCE0)
    PRELU_KERNELS_2D(I16,  F16,  F16,  _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(U8,   F16,  U8,   _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(U8,   F16,  F16,  _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(I8,   F16,  I8,   _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(I8,   F16,  I8,   _2D_OPT, KERNEL_SOURCE0)
    PRELU_KERNELS_2D(I8,   F16,  F16,  _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(U8,   U8,   U8,   _2D,     KERNEL_SOURCE0)
    PRELU_KERNELS_2D(U8,   U8,   F16,  _2D,     KERNEL_SOURCE0)
};

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

DEF_KERNEL_INITIALIZER(_prelu_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    int8_t      in0_fl          = 0;
    int32_t     inputZP0        = 0;
    float       input_scale0    = 1.0f;
    int32_t     inputZP1        = 0;
    float       input_scale1    = 1.0f;
    int8_t      out_fl          = 0;
    float       outputZP        = 0;

    int32_t  shift0             = 0;
    vsi_bool is_ge_fl           = FALSE;
    vsi_bool is_2d_img          = FALSE;
    uint32_t evis_version       = 0;

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    vsi_size_array_t * out_shape = NULL;
    uint32_t pack_key;
    vx_context                  ctx       = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_t   hw_param;

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_t));
    status = vxQueryHardwareCaps(ctx, &hw_param, sizeof(vx_hardware_caps_params_t));
    CHECK_STATUS_FAIL_GOTO(status, final);

    if (hw_param.evis1 == TRUE && hw_param.evis2 == FALSE)
    {
        evis_version = 1;
    }
    else if (hw_param.evis1 == FALSE && hw_param.evis2 == TRUE)
    {
        evis_version = 2;
    }

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_shape  = attr[2]->shape;
    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        in0_fl = (int8_t)attr[0]->dfp.fl;
        if (in0_fl >= 0)
        {
            input_scale0 = 1.0f / (vx_float32) ((int64_t)1 << in0_fl);
        }
        else if (in0_fl < 0)
        {
            input_scale0 = (vx_float32) ((int64_t)1 << -in0_fl);
        }
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        inputZP0    = attr[0]->asymm.zero_point;
        input_scale0  = attr[0]->asymm.scale;
    }

    if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        inputZP1 = attr[1]->asymm.zero_point;
        input_scale1  = attr[1]->asymm.scale;
    }

    if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        out_fl = (int8_t)attr[2]->dfp.fl;

        if (out_fl >= 0)
            input_scale0 *= (vx_float32)((int64_t)1 << out_fl);
        else if (out_fl < 0)
            input_scale0 *= 1.0f / (vx_float32) ((int64_t)1 << -out_fl);
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        out_fl = 1;
        outputZP      = (float)attr[2]->asymm.zero_point;
        input_scale0   = input_scale0 / attr[2]->asymm.scale;
    }
    shift0 = in0_fl - out_fl;

    is_2d_img = (out_shape->size < 3) || (out_shape->data[2] == 1);
    is_ge_fl  = shift0 >= 0;

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE, GE_FL, IMG_2D, EVIS2 )    \
        (IN0_TYPE  | ( OUT_TYPE << 16) | (GE_FL << 24) | (IMG_2D << 25) | (EVIS2 << 26))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[2]->dtype, is_ge_fl, is_2d_img, evis_version );

    if ( attr[0]->dtype == I8 && attr[2]->dtype == I8 && is_ge_fl)
    {
        gpu_param.global_scale[0] = 16;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }
    else
    {
        gpu_param.global_scale[0] = 8;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }

    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    switch( pack_key )
    {
        case _PACK_SELECT_KEY( I8,  I8,  1, 1, 2 ):
        case _PACK_SELECT_KEY( I16, I16, 1, 1, 2 ):
        {
            gpu_dp_inst_t uniPreluDFPLo_2x8b = {{
                0x77777777, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0x00000000, // BSelt
                0x30201000, 0x70605040, // BBin
                0x00004000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniPreluDFPHi_2x8b = {{
                0x77777777, // TCfg
                0x44444444, // ASelt
                0xbbaa9988, 0xffeeddcc, // ABin
                0x00000000, // BSelt
                0x30201000, 0x70605040, // BBin
                0x00004000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            if ( attr[0]->dtype == I16 )
            {
                uniPreluDFPLo_2x8b.data[7] = 0x00003000;
                uniPreluDFPHi_2x8b.data[7] = 0x00003000;
            }

            gpu_dp_inst_update_postshfit( &uniPreluDFPLo_2x8b, shift0 );
            gpu_dp_inst_update_postshfit( &uniPreluDFPHi_2x8b, shift0 );

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniPreluDFPLo_2x8b", &uniPreluDFPLo_2x8b );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniPreluDFPHi_2x8b", &uniPreluDFPHi_2x8b );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
        case _PACK_SELECT_KEY( I8,  I8,  1, 1, 1 ):
        case _PACK_SELECT_KEY( I16, I16, 1, 1, 1 ):
        {
            gpu_dp_inst_t uniPreluInt8_2x8 = {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0xb3a29180, 0xf7e6d5c4, // ABin
                0x66666666, // BSelt
                0x30201000, 0x70605040, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniPreluInt16_part0_4x4 = {{
                0x05050505, // TCfg
                0x00000000, // ASelt
                0x00510040, 0x00730062, // ABin
                0x06060606, // BSelt
                0x00100000, 0x00300020, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniPreluInt16_part1_4x4 = {{
                0x05050505, // TCfg
                0x00000000, // ASelt
                0x00510040, 0x00730062, // ABin
                0x06060606, // BSelt
                0x00500040, 0x00700060, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            gpu_dp_inst_update_postshfit( &uniPreluInt8_2x8, shift0 );
            gpu_dp_inst_update_postshfit( &uniPreluInt16_part0_4x4, shift0 );
            gpu_dp_inst_update_postshfit( &uniPreluInt16_part1_4x4, shift0 );

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniPreluInt8_2x8", &uniPreluInt8_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniPreluInt16_part0_4x4", &uniPreluInt16_part0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniPreluInt16_part1_4x4", &uniPreluInt16_part1_4x4 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
        case _PACK_SELECT_KEY( BF16, BF16, 1, 1, 1 ):
        case _PACK_SELECT_KEY( BF16, BF16, 1, 1, 2 ):
        case _PACK_SELECT_KEY( BF16, BF16, 1, 0, 1 ):
        case _PACK_SELECT_KEY( BF16, BF16, 1, 0, 2 ):
        {
            gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x01010000, 0x03030202, // ABin
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
            gpu_dp_inst_t uniConvF16toF32_Part0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConvF16toF32_Part1_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniPackedBF16_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x07050301, 0x07050301, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
            if (attr[1]->dtype == F16)
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvF16toF32_Part0_4x4", &uniConvF16toF32_Part0_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvF16toF32_Part1_4x4", &uniConvF16toF32_Part1_4x4 );
            }
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniPackedBF16_2x8", &uniPackedBF16_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        {
            gpu_dp_inst_t uniConvF16toF32_part0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConvF16toF32_part1_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtractInteger_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtractHalf8_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniDataSubZPtoFp32Part0_4x4 = {{
                0x09090909, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniDataSubZPtoFp32Part1_4x4 = {{
                0x09090909, // TCfg
                0x04040404, // ASelt
                0x00050004, 0x00070006, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                "uniDataSubZPtoFp32Part0_4x4", &uniDataSubZPtoFp32Part0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniDataSubZPtoFp32Part1_4x4", &uniDataSubZPtoFp32Part1_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniConvF16toF32_part0_4x4", &uniConvF16toF32_part0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniConvF16toF32_part1_4x4", &uniConvF16toF32_part1_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "inputZP0", &inputZP0 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "input_scale0", &input_scale0 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "inputZP1", &inputZP1 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "input_scale1", &input_scale1 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "outputZP", &outputZP );
            if (attr[2]->dtype == F16)
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtact8Bin_2x8", &uniExtractHalf8_2x8 );
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtact8Bin_2x8", &uniExtractInteger_2x8 );
            }
        }
        break;
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
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
} /* _prelu_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype;
    vsi_nn_kernel_dtype_e input1_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    int8_t input_fl = inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP ?
        inputs[0]->attr.dtype.fl : 0;
    int8_t output_fl = outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP ?
        outputs[0]->attr.dtype.fl : 1;
    vsi_nn_shader_type_e  sh_type = image_2d ? (input_fl >= output_fl ? _2D_OPT : _2D) : _3D;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_PRELU_KEY( input0_dtype, input1_dtype, output_dtype, sh_type );

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _prelu_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    vsi_size_t new_rank = 0;
    vsi_bool ret;
    int32_t is_per_channel_alpha = 0;

    is_per_channel_alpha = vsi_nn_kernel_param_get_int32(params, "is_per_channel_alpha");

    if (is_per_channel_alpha)
    {
        return NULL;
    }

    ret = vsi_nn_kernel_optimize_eltwise_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            inputs[1]->attr.size, inputs[1]->attr.dim_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    if (ret)
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shapes[0], (uint32_t)new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
                inputs[1], shapes[1], (uint32_t)new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
                outputs[0], shapes[2], (uint32_t)new_rank );
    }
    else
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[2]->attr.size,
                reshape_tensors[2]->attr.dim_num ) )
    {
        goto final;
    }

    // Reorder tensor
    image_2d = (reshape_tensors[2]->attr.dim_num == 2);
    status = _query_kernel( reshape_tensors, &reshape_tensors[2], image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_PARAM_NUM,
                    reshape_tensors, 2, &reshape_tensors[2], 1 );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_PARAM_NUM );
        }
    }

final:
    vsi_nn_ReleaseTensor( &reshape_tensors[0] );
    vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    vsi_nn_ReleaseTensor( &reshape_tensors[2] );

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( prelu, _setup )

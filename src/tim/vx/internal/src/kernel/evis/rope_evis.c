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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
    B---batch
    N---num_heads
    S---sequence length
    H---head size
 */
typedef enum
{
    LAYOUT_NONE,
    LAYOUT_BNHS,
    LAYOUT_BNH1,
    LAYOUT_BSNH,
    LAYOUT_BNSH,
} _internal_rope_layout_e;

// Add kernel hashtable here
#define STR(a) #a
#define ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, LAYOUT, INTERLEAVED ) \
      ((IN0_DTYPE) | (IN1_DTYPE << 8) | (OUT_DTYPE << 16) | (LAYOUT << 24) | (INTERLEAVED << 28))
#define PACK_KERNEL_BNHS_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, LAYOUT_BNHS, 0 ), \
         CVIVANTE_NAMESPACE("evis.rope_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_bnhs"), \
         "rope_0" }
#define PACK_KERNEL_BNH1_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, LAYOUT_BNH1, 0 ), \
         CVIVANTE_NAMESPACE("evis.rope_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_bnh1"), \
         "rope_1" }

#define PACK_KERNEL_BSNH_INTERLEVEAD_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, LAYOUT_BSNH, 1 ), \
         CVIVANTE_NAMESPACE("evis.rope_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_bsnh"), \
         "rope_2" }

#define PACK_KERNEL_BNSH_INTERLEVEAD_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, LAYOUT_BNSH, 1 ), \
         CVIVANTE_NAMESPACE("evis.rope_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_bnsh"), \
         "rope_3" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

#define PACK_KERNEL_MAP(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    PACK_KERNEL_BNHS_MAP(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
    PACK_KERNEL_BNH1_MAP(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
    PACK_KERNEL_BSNH_INTERLEVEAD_MAP(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
    PACK_KERNEL_BNSH_INTERLEVEAD_MAP(IN0_TYPE, IN1_TYPE, OUT_TYPE),

static const _kernel_map_type _rope_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( BF16, BF16, BF16)
    PACK_KERNEL_MAP( F16,  F16,  F16 )
    PACK_KERNEL_MAP( I16,  I16,  I16 )
    PACK_KERNEL_MAP( I16,  F16,  I16 )
    PACK_KERNEL_MAP( I16,  I16,  I8 )
    PACK_KERNEL_MAP( I16,  F16,  I8 )
    PACK_KERNEL_MAP( I16,  I16,  U8 )
    PACK_KERNEL_MAP( I16,  F16,  U8 )
    PACK_KERNEL_MAP( U16,  U16,  U16 )
    PACK_KERNEL_MAP( U16,  F16,  U16 )
    PACK_KERNEL_MAP( I8,   I8,   I8  )
    PACK_KERNEL_MAP( I8,   F16,  I8  )
    PACK_KERNEL_MAP( U8,   U8,   U8  )
    PACK_KERNEL_MAP( U8,   F16,  U8  )
};

/*
 * Kernel params
 */
static vx_param_description_t _rope_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _ROPE_PARAM_NUM  _cnt_of_array( _rope_kernel_param_def )
#define SCALAR_AXIS       (4)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_rope_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
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
    vsi_nn_kernel_tensor_attr_t* out_attr = NULL;
    vsi_nn_kernel_tensor_attr_t* in0_attr = NULL;
    vsi_nn_kernel_tensor_attr_t* in1_attr = NULL;
    vsi_nn_kernel_tensor_attr_t* in2_attr = NULL;
    vsi_size_array_t* in_shape = NULL;
    vsi_nn_kernel_dtype_e in0_dtype = F16;
    vsi_nn_kernel_dtype_e in1_dtype = F16;
    vsi_nn_kernel_dtype_e in2_dtype = F16;
    vsi_nn_kernel_dtype_e out_dtype = F16;
    float in0_scale = 1.0f;
    float in1_scale = 1.0f;
    float in2_scale = 1.0f;
    float output_scale = 1.0f;
    float output_zp = 0;
    int32_t in0_zp = 0;
    int32_t cos_zp = 0;
    int32_t sin_zp = 0;
    int32_t p = 0;
    int32_t axis = 0;
    int32_t interleaved = 0;
    int32_t half_head_size = 1;
    vsi_size_t shape[3] = {1};
    uint32_t pack_key = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param);
    VSI_UNREFERENCED(param_size);
    // Add initializer

    in0_attr = vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[0]);
    CHECK_PTR_FAIL_GOTO(in0_attr, "Create tensor attr buffer fail.", final);
    in1_attr = vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[1]);
    CHECK_PTR_FAIL_GOTO(in1_attr, "Create tensor attr buffer fail.", final);
    in2_attr = vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[2]);
    CHECK_PTR_FAIL_GOTO(in2_attr, "Create tensor attr buffer fail.", final);
    out_attr = vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[3]);
    CHECK_PTR_FAIL_GOTO(out_attr, "Create tensor attr buffer fail.", final);

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &p);
    CHECK_STATUS_FAIL_GOTO(status, final);

    axis = p & 0xFFFF;
    interleaved = (p >> 16) & 0xFFFF;

    in_shape = in0_attr->shape;
    in0_dtype = in0_attr->dtype;
    in1_dtype = in1_attr->dtype;
    in2_dtype = in2_attr->dtype;
    out_dtype = out_attr->dtype;

    in0_scale = in0_attr->scale;
    in1_scale = in1_attr->scale;
    in2_scale = in2_attr->scale;
    in0_zp = -in0_attr->zero_point;
    cos_zp = -in1_attr->zero_point;
    sin_zp = -in2_attr->zero_point;
    output_scale = out_attr->scale;
    output_zp = (float)out_attr->zero_point;

    half_head_size = (int32_t)(in_shape->data[axis] / 2);
    shape[0] = in_shape->data[0];
    shape[1] = in_shape->data[1];
    shape[2] = in_shape->data[2];
    shape[axis] = half_head_size;

    gpu_param.global_scale[0] = 8;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = gpu_align_p2((shape[0] + \
        gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = shape[1];
    gpu_param.global_size[2] = shape[2];

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE )    \
        ((IN0_TYPE) | (IN1_TYPE << 8) | (IN2_TYPE << 16) | (OUT_TYPE << 24))

    pack_key = _PACK_SELECT_KEY(in0_dtype, in1_dtype, in2_dtype, out_dtype);
    switch (pack_key)
    {
    case _PACK_SELECT_KEY(BF16, BF16, BF16, BF16):
    {
        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = { {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = { {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractOddData_2x8 = { {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        if (interleaved && axis == 0)
        {
            uniExtractOddData_2x8.data[1] = 0x10101010;
            uniExtractOddData_2x8.data[2] = 0x03030101;
            uniExtractOddData_2x8.data[3] = 0x07070505;
        }
        else
        {
            status = vsi_nn_kernel_gpu_add_param(node,
                "half_head_size", &half_head_size);
            CHECK_STATUS_FAIL_GOTO(status, final);
        }
        status = vsi_nn_kernel_gpu_add_param(node,
            "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node,
            "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node,
            "uniExtractOddData_2x8", &uniExtractOddData_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final);
    }
    break;
    case _PACK_SELECT_KEY(I16, I16, I16, I16):
    case _PACK_SELECT_KEY(I16, F16, F16, I16):
    case _PACK_SELECT_KEY(I16, I16, I16, I8):
    case _PACK_SELECT_KEY(I16, F16, F16, I8):
    case _PACK_SELECT_KEY(I16, I16, I16, U8):
    case _PACK_SELECT_KEY(I16, F16, F16, U8):
    case _PACK_SELECT_KEY(F16, F16, F16, F16):
        {
            float scale0 = in0_scale * in1_scale / output_scale;
            float scale1 = in0_scale* in2_scale / output_scale;
            gpu_dp_inst_t uniExtractHalf8_2x8 = { {
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtractInteger_2x8 = { {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniATimesB_0_4x4 = { {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x01010101, // BSelt
                0x00010000, 0x00030002, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniATimesB_1_4x4 = { {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x01010101, // BSelt
                0x00050004, 0x00070006, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAEvenTimesB_0_4x4 = { {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00020000, 0x00060004, // ABin
                0x01010101, // BSelt
                0x00010000, 0x00030002, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAEvenTimesB_1_4x4 = { {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00020000, 0x00060004, // ABin
                0x01010101, // BSelt
                0x00050004, 0x00070006, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAOddTimesB_0_4x4 = { {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00030001, 0x00070005, // ABin
                0x01010101, // BSelt
                0x00010000, 0x00030002, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAOddTimesB_1_4x4 = { {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00030001, 0x00070005, // ABin
                0x01010101, // BSelt
                0x00050004, 0x00070006, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            if (interleaved && axis == 0)
            {
                uniExtractHalf8_2x8.data[1] = 0x10101010;
                uniExtractHalf8_2x8.data[2] = 0x02020000;
                uniExtractHalf8_2x8.data[3] = 0x06060404;
                uniExtractInteger_2x8.data[1] = 0x10101010;
                uniExtractInteger_2x8.data[2] = 0x01010000;
                uniExtractInteger_2x8.data[3] = 0x03030202;

                status = vsi_nn_kernel_gpu_add_param(node,
                    "uniAEvenTimesB_0_4x4", &uniAEvenTimesB_0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniAEvenTimesB_1_4x4", &uniAEvenTimesB_1_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniAOddTimesB_0_4x4", &uniAOddTimesB_0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniAOddTimesB_1_4x4", &uniAOddTimesB_1_4x4);
            }
            else
            {
                status = vsi_nn_kernel_gpu_add_param(node,
                    "uniATimesB_0_4x4", &uniATimesB_0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniATimesB_1_4x4", &uniATimesB_1_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "half_head_size", &half_head_size);
            }
            status |= vsi_nn_kernel_gpu_add_param(node,
                "scale0", &scale0);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "scale1", &scale1);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "output_zp", &output_zp);
            if (out_dtype == F16)
            {
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniExtract8Data_2x8", &uniExtractHalf8_2x8);
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniExtract8Data_2x8", &uniExtractInteger_2x8);
            }
            CHECK_STATUS_FAIL_GOTO(status, final);
        }
        break;
    case _PACK_SELECT_KEY(I8,  I8,  I8,  I8):
    case _PACK_SELECT_KEY(U8,  U8,  U8,  U8):
    case _PACK_SELECT_KEY(U16, U16, U16, U16):
    case _PACK_SELECT_KEY(I8,  F16, F16, I8):
    case _PACK_SELECT_KEY(U8,  F16, F16, U8):
    case _PACK_SELECT_KEY(U16, F16, F16, U16):
        {
            float scale0 = in0_scale * in1_scale / output_scale;
            float scale1 = in0_scale* in2_scale / output_scale;
            gpu_dp_inst_t uniExtractInteger_2x8 = { {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAMinusZp_0_4x4 = { {
                0x0d0d0d0d, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAMinusZp_1_4x4 = { {
                0x0d0d0d0d, // TCfg
                0x04040404, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAEvenMinusZp_4x4 = { {
                0x0d0d0d0d, // TCfg
                0x04040404, // ASelt
                0x00020000, 0x00060004, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniAOddMinusZp_4x4 = { {
                0x0d0d0d0d, // TCfg
                0x04040404, // ASelt
                0x00030001, 0x00070005, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            if (interleaved && axis == 0)
            {
                uniExtractInteger_2x8.data[1] = 0x10101010;
                uniExtractInteger_2x8.data[2] = 0x01010000;
                uniExtractInteger_2x8.data[3] = 0x03030202;

                status = vsi_nn_kernel_gpu_add_param(node,
                    "uniAEvenMinusZp_4x4", &uniAEvenMinusZp_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node,
                    "uniAOddMinusZp_4x4", &uniAOddMinusZp_4x4);
            }
            else
            {
                status = vsi_nn_kernel_gpu_add_param(node,
                    "half_head_size", &half_head_size);
            }
            status |= vsi_nn_kernel_gpu_add_param(node,
                "uniAMinusZp_0_4x4", &uniAMinusZp_0_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "uniAMinusZp_1_4x4", &uniAMinusZp_1_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "scale0", &scale0);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "scale1", &scale1);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "output_zp", &output_zp);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "in0_zp", &in0_zp);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "cos_zp", &cos_zp);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "sin_zp", &sin_zp);
            status |= vsi_nn_kernel_gpu_add_param(node,
                "uniExtract8Data_2x8", &uniExtractInteger_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final);
        }
        break;
    default:
        break;
    }
    status = vsi_nn_kernel_gpu_config(node, &gpu_param);
final:
    if (in0_attr) vsi_nn_kernel_tensor_attr_release(&in0_attr);
    if (in1_attr) vsi_nn_kernel_tensor_attr_release(&in1_attr);
    if (in2_attr) vsi_nn_kernel_tensor_attr_release(&in2_attr);
    if (out_attr) vsi_nn_kernel_tensor_attr_release(&out_attr);
    return status;
} /* _rope_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis,
    int32_t interleaved,
    _internal_rope_layout_e *layout
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    int32_t in0_zp = vsi_nn_get_tensor_zero_point(inputs[0]);
    int32_t in1_zp = vsi_nn_get_tensor_zero_point(inputs[1]);
    int32_t in2_zp = vsi_nn_get_tensor_zero_point(inputs[2]);
    const _kernel_map_type * kernel_map = _rope_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _rope_kernel_map );
    vx_param_description_t * param_def  = _rope_kernel_param_def;
    vx_kernel_initialize_f  initializer = _rope_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    in2_dtype  = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    /*only support symmetric int16*/
    if ( ( (in0_dtype == I16 && in1_dtype == I16 && out_dtype == I16) ||
           (in0_dtype == I16 && in1_dtype == F16 && out_dtype == I16) ||
           (in0_dtype == I16 && in1_dtype == F16 && out_dtype == I8)  ||
           (in0_dtype == I16 && in1_dtype == I16 && out_dtype == I8)  ||
           (in0_dtype == I16 && in1_dtype == F16 && out_dtype == U8)  ||
           (in0_dtype == I16 && in1_dtype == I16 && out_dtype == U8) ) &&
        (in0_zp != 0 || in1_zp != 0 || in2_zp != 0))
    {
        return VSI_FAILURE;
    }

    if (axis == 1 && inputs[0]->attr.size[0] == inputs[1]->attr.size[0] &&
        in1_dtype == in2_dtype)
    {
        if (inputs[0]->attr.size[0] == 1)
        {
            *layout = LAYOUT_BNH1;
        }
        else
        {
            *layout = LAYOUT_BNHS;
        }
    }
    else if (axis == 0 && in1_dtype == in2_dtype)
    {
        if (inputs[0]->attr.size[2] == inputs[1]->attr.size[2] &&
            inputs[1]->attr.size[1] == 1)
        {
            *layout = LAYOUT_BSNH;
        }
        else
        {
            *layout = LAYOUT_BNSH;
        }
    }

    key = ROPE_HASH_KEY(in0_dtype, in1_dtype, out_dtype, *layout, interleaved);

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < (uint32_t)kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = _cnt_of_array( _rope_kernel_param_def );
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
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
    vsi_nn_kernel_node_param_t node_params[_ROPE_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = 0;
    int32_t i = 0;
    int32_t interleaved = 0;
    int32_t param = 0;
    vsi_size_t shape[3][VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_tensor_t* rs_tensors[4] = { NULL };
    vsi_nn_tensor_t* reshape_tensors[4] = { NULL };
    _internal_rope_layout_e layout = LAYOUT_NONE;

    VSI_UNREFERENCED(params);

    axis = vsi_nn_kernel_param_get_int32(params, "axis");
    interleaved = vsi_nn_kernel_param_get_int32(params, "interleaved");

    // Check if gpu can support the size
    if ( !vsi_nn_kernel_gpu_check_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, axis, interleaved, &layout );
    if (outputs[0]->attr.size[0] == 1 || layout == LAYOUT_BSNH)
    {
        memcpy(shape[0], inputs[0]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        memcpy(shape[1], inputs[1]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        memcpy(shape[2], outputs[0]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));

        if (outputs[0]->attr.size[0] == 1)
        {
            for (i = 1; i < 3; i++)
            {
                shape[0][i - 1] = shape[0][i];
                shape[1][i - 1] = shape[1][i];
                shape[2][i - 1] = shape[2][i];
            }
            shape[0][2] = 1;
            shape[1][2] = 1;
            shape[2][2] = 1;
        }
        else
        {
            int32_t j = 0;
            for (i = 0; i < 3; i++)
            {
                if (shape[1][i] != 1)
                {
                    shape[1][j] = shape[1][i];
                    j ++;
                }
            }
            for (; j < 3; j++)
            {
                shape[1][j] = 1;
            }
        }

        rs_tensors[0] = vsi_nn_reshape_tensor(graph,
            inputs[0], shape[0], inputs[0]->attr.dim_num);
        rs_tensors[1] = vsi_nn_reshape_tensor(graph,
            inputs[1], shape[1], inputs[1]->attr.dim_num);
        rs_tensors[2] = vsi_nn_reshape_tensor(graph,
            inputs[2], shape[1], inputs[2]->attr.dim_num);
        rs_tensors[3] = vsi_nn_reshape_tensor(graph,
            outputs[0], shape[2], outputs[0]->attr.dim_num);

        if (outputs[0]->attr.size[0] == 1 && axis > 0)
        {
            axis--;
        }
        reshape_tensors[0] = rs_tensors[0];
        reshape_tensors[1] = rs_tensors[1];
        reshape_tensors[2] = rs_tensors[2];
        reshape_tensors[3] = rs_tensors[3];
    }
    else
    {
        reshape_tensors[0] = inputs[0];
        reshape_tensors[1] = inputs[1];
        reshape_tensors[2] = inputs[2];
        reshape_tensors[3] = outputs[0];
    }

    param = (interleaved << 16) | axis;
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ROPE_PARAM_NUM,
                reshape_tensors, input_num, &reshape_tensors[3], output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_AXIS] = vsi_nn_kernel_scalar_create(graph, I32, &param);
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ROPE_PARAM_NUM );
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_AXIS]);
        }
    }

    for (i = 0; i < 4; i++)
    {
        vsi_safe_release_tensor(rs_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( rope, _setup )


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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_BILINEAR_GRID_SAMPLE,
} _internal_kernel_e;

#define STR(a) #a

#define _BILINEAR_GRID_SAMPLE_KERNEL_SOURCE(_input_type, _output_type) \
    "bilinear_grid_sample_" #_input_type "_to_" #_output_type

// Add kernel hashtable here
#define BILINEAR_GRID_SAMPLE_HASH_KEY(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        ((IN1_DTYPE << 20) | (IN0_DTYPE << 8) | (OUT_DTYPE))
#define PACK_KERNEL_MAP(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        {                                                                   \
        BILINEAR_GRID_SAMPLE_HASH_KEY(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE), \
            CVIVANTE_NAMESPACE("evis.bilinear_grid_sample_" STR(IN0_DTYPE) "_" STR(IN1_DTYPE) "to" STR(OUT_DTYPE)), \
            _BILINEAR_GRID_SAMPLE_KERNEL_SOURCE(IN0_DTYPE, OUT_DTYPE)     \
        }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _bilinear_grid_sample_kernel_map[] =
{
    PACK_KERNEL_MAP(F16,  F32,  F16),
    PACK_KERNEL_MAP(F16,  U8,   F16),
    PACK_KERNEL_MAP(F16,  F16,  F16),
    PACK_KERNEL_MAP(F16,  F32,  U8),
    PACK_KERNEL_MAP(F16,  F16,  U8),
    PACK_KERNEL_MAP(F16,  U8,   U8),
    PACK_KERNEL_MAP(U8,   U8,   U8),
    PACK_KERNEL_MAP(U8,   F16,  U8),
    PACK_KERNEL_MAP(U8,   F32,  U8),
    PACK_KERNEL_MAP(I16,  I16,  I16),
    PACK_KERNEL_MAP(I8,   I8,   I8),
    PACK_KERNEL_MAP(BF16, BF16, BF16),
};


/*
 * Kernel params
 */
static vx_param_description_t _bilinear_grid_sample_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _BILINEAR_GRID_SAMPLE_PARAM_NUM  _cnt_of_array( _bilinear_grid_sample_kernel_param_def )

#define SCALAR_ALIGN_CORNERS (3)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_bilinear_grid_sample_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {3, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    vsi_nn_kernel_tensor_attr_t* output_attr = NULL;
    vsi_nn_kernel_tensor_attr_t* input_attr[2] = {NULL};
    vsi_size_array_t* out_shape = NULL;
    vsi_size_array_t* in0_shape = NULL;
    vsi_nn_kernel_dtype_e input0_dtype = F16;
    vsi_nn_kernel_dtype_e input1_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = F16;

    uint32_t depth = 0;
    float half_input0_wh[2];
    float add_float_value[2];
    uint32_t in0_width;
    uint32_t in0_height;
    uint32_t out_width;
    uint32_t out_height;
    int32_t align_corners;

    int32_t src0FixPointPos = 0;
    int32_t src1FixPointPos = 0;
    int32_t dstFixPointPos  = 0;
    float   input0_scale    = 1.0;
    int32_t input0ZP        = 0;
    float   input1_scale    = 1.0;
    int32_t input1ZP        = 0;
    float   output_scale    = 1.0;
    int32_t outputZP        = 0;

    input_attr[0] =
        vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[0]);
    CHECK_PTR_FAIL_GOTO(
        input_attr[0], "Create tensor attr buffer fail.", final);

    input_attr[1] =
        vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[1]);
    CHECK_PTR_FAIL_GOTO(
        input_attr[1], "Create tensor attr buffer fail.", final);

    output_attr =
        vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[2]);
    CHECK_PTR_FAIL_GOTO(output_attr, "Create tensor attr buffer fail.", final);

   status = vsi_nn_kernel_scalar_read_int32(
        (vsi_nn_kernel_scalar_t)param[SCALAR_ALIGN_CORNERS], &(align_corners));
    CHECK_STATUS_FAIL_GOTO(status, final);

    out_shape = output_attr->shape;
    in0_shape = input_attr[0]->shape;
    input0_dtype = input_attr[0]->dtype;
    input1_dtype = input_attr[1]->dtype;
    output_dtype = output_attr->dtype;

    if (U8 == input0_dtype && VSI_NN_KERNEL_QUANT_ASYMM == input_attr[0]->quant) {
        input0_scale = input_attr[0]->asymm.scale;
        input0ZP     = input_attr[0]->asymm.zero_point;
    } else if (VSI_NN_KERNEL_QUANT_DFP == input_attr[0]->quant) {
        src0FixPointPos = input_attr[0]->dfp.fl;
        if (src0FixPointPos >= 0) {
            input0_scale = 1.0f / (float)((int64_t)1 << src0FixPointPos);
        } else if (src0FixPointPos < 0) {
            input0_scale = (float)((int64_t)1 << -src0FixPointPos);
        }
        input0ZP = 0;
    } else {
        input0_scale = 1.0f;
        input0ZP     = 0;
    }

    if (U8 == input1_dtype && VSI_NN_KERNEL_QUANT_ASYMM == input_attr[1]->quant) {
        input1_scale = input_attr[1]->asymm.scale;
        input1ZP     = input_attr[1]->asymm.zero_point;
    } else if (VSI_NN_KERNEL_QUANT_DFP == input_attr[1]->quant) {
        src1FixPointPos = input_attr[1]->dfp.fl;
        if (src1FixPointPos >= 0) {
            input1_scale = 1.0f / (float)((int64_t)1 << src1FixPointPos);
        } else if (src1FixPointPos < 0) {
            input1_scale = (float)((int64_t)1 << -src1FixPointPos);
        }
        input1ZP = 0;
    } else {
        input1_scale = 1.0f;
        input1ZP     = 0;
    }

    if (U8 == output_dtype && VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant) {
        output_scale = output_attr->asymm.scale;
        outputZP = output_attr->asymm.zero_point;
    } else if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant) {
        dstFixPointPos = output_attr->dfp.fl;
        if (dstFixPointPos >= 0) {
            output_scale = (float)((int64_t)1 << dstFixPointPos);
        } else if (dstFixPointPos < 0) {
            output_scale = 1.0f / (float)((int64_t)1 << -dstFixPointPos);
        }
        outputZP = 0;
    } else {
        output_scale = 1.0;
        outputZP = 0;
    }


    in0_width  = (uint32_t)(in0_shape->data[0]);
    in0_height = (uint32_t)(in0_shape->data[1]);
    depth      = (uint32_t)(in0_shape->data[2]);
    out_width  = (uint32_t)(out_shape->data[0]);
    out_height = (uint32_t)(out_shape->data[1]);

    if (align_corners) {
        half_input0_wh[0]  = ((float)in0_width - 1.0f) * 0.5f;
        half_input0_wh[1]  = ((float)in0_height - 1.0f) * 0.5f;
        add_float_value[0] = half_input0_wh[0];
        add_float_value[1] = half_input0_wh[1];
    } else {
        half_input0_wh[0]  = (float)in0_width * 0.5f;
        half_input0_wh[1]  = (float)in0_height * 0.5f;
        add_float_value[0] = half_input0_wh[0] - 0.5f;
        add_float_value[1] = half_input0_wh[1] - 0.5f;
    }

    status  = vsi_nn_kernel_gpu_add_param(node, "half_input0_wh", half_input0_wh);
    status |= vsi_nn_kernel_gpu_add_param(node, "add_float_value", add_float_value);
    status |= vsi_nn_kernel_gpu_add_param(node, "depth", &depth);

    {
        gpu_dp_inst_t uniFp16toFp32_part0_4x4 = {
            {
                0x01010101,  // TCfg
                0x00000000,  // ASelt
                0x00010000, 0x00030002,  // ABin
                0x02020202,  // BSelt
                0x00000000, 0x00000000,  // BBin
                0x00000400,  // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000  // Constant
            },
            GPU_DP_TYPE_16};
        gpu_dp_inst_t uniFp16toFp32_part1_4x4 = {
            {
                0x01010101,  // TCfg
                0x00000000,  // ASelt
                0x00050004, 0x00070006,  // ABin
                0x02020202,  // BSelt
                0x00000000, 0x00000000,  // BBin
                0x00000400,  // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000  // Constant
            },
            GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8SubZPtoFp32_part0_4x4 = {
            {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8SubZPtoFp32_part1_4x4 = {
            {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniEvenBintoFp32_4x4 = {
            {
                0x01010101,  // TCfg
                0x00000000,  // ASelt
                0x00020000, 0x00060004,  // ABin
                0x02020202,  // BSelt
                0x00000000, 0x00000000,  // BBin
                0x00000100,  // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000  // Constant
            }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniOddSubEvenBin_4x4 = {
            {
                0x09090909,  // TCfg
                0x00000000,  // ASelt
                0x00230001, 0x00670045,  // ABin
                0x0a0a0a0a,  // BSelt
                0x00000000, 0x00000000,  // BBin
                0x00000100,  // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000  // Constant
            }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtactHalf8_2x8 = {
            {
                0x11111111,  // TCfg
                0x11110000,  // ASelt
                0x06040200, 0x06040200,  // ABin
                0x22222222,  // BSelt
                0x00000000, 0x00000000,  // BBin
                0x00000100,  // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00  // Constant
            }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniExtact8Bit_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        if (F16 == input0_dtype &&
            (F16 == input1_dtype || F32 == input1_dtype ||
             U8 == input1_dtype) &&
            F16 == output_dtype) {
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniEvenBintoFp32_4x4", &uniEvenBintoFp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniOddSubEvenBin_4x4", &uniOddSubEvenBin_4x4);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniExtactHalf8_2x8", &uniExtactHalf8_2x8);
            if (F16 == input1_dtype) {
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniFp16toFp32_part0_4x4", &uniFp16toFp32_part0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniFp16toFp32_part1_4x4", &uniFp16toFp32_part1_4x4);
            } else if (U8 == input1_dtype) {
                status |=
                    vsi_nn_kernel_gpu_add_param(node, "input1_ZP", &input1ZP);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "input1Scale", &input1_scale);
                status |=
                    vsi_nn_kernel_gpu_add_param(node,
                                                "uniU8SubZPtoFp32_part0_4x4",
                                                &uniU8SubZPtoFp32_part0_4x4);
                status |=
                    vsi_nn_kernel_gpu_add_param(node,
                                                "uniU8SubZPtoFp32_part1_4x4",
                                                &uniU8SubZPtoFp32_part1_4x4);
            }
        } else if (F16 == input0_dtype &&
                   (F16 == input1_dtype || F32 == input1_dtype ||
                    U8 == input1_dtype) &&
                   U8 == output_dtype) {
            float uint8Scale = 1.0f / output_scale;
            float uint8ZP_out = (float)outputZP;
            status |= vsi_nn_kernel_gpu_add_param(node, "uint8Scale", &uint8Scale);
            status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP", &uint8ZP_out);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniEvenBintoFp32_4x4", &uniEvenBintoFp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniOddSubEvenBin_4x4", &uniOddSubEvenBin_4x4);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
            if (U8 == input1_dtype) {
                status |=
                    vsi_nn_kernel_gpu_add_param(node, "input1_ZP", &input1ZP);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "input1Scale", &input1_scale);
                status |=
                    vsi_nn_kernel_gpu_add_param(node,
                                                "uniU8SubZPtoFp32_part0_4x4",
                                                &uniU8SubZPtoFp32_part0_4x4);
                status |=
                    vsi_nn_kernel_gpu_add_param(node,
                                                "uniU8SubZPtoFp32_part1_4x4",
                                                &uniU8SubZPtoFp32_part1_4x4);
            } else if (F16 == input1_dtype) {
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniFp16toFp32_part0_4x4", &uniFp16toFp32_part0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniFp16toFp32_part1_4x4", &uniFp16toFp32_part1_4x4);
            }
        }
        else if (U8 == input0_dtype &&
                   (F16 == input1_dtype || F32 == input1_dtype ||
                    U8 == input1_dtype) &&
                 U8 == output_dtype) {
            float uint8Scale  = input0_scale / output_scale;
            float uint8ZP_out = (float)outputZP;
            gpu_dp_inst_t uniU8SubZPtoFp32_left_4x4 = {{
                0x09090909, // TCfg
                0x04040404, // ASelt
                0x00020000, 0x00060004, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8RightSubLeft_4x4 = {{
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00230001, 0x00670045, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP", &input0ZP);
            status |= vsi_nn_kernel_gpu_add_param(node, "uint8Scale", &uint8Scale);
            status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP", &uint8ZP_out);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZPtoFp32_left_4x4", &uniU8SubZPtoFp32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8RightSubLeft_4x4", &uniU8RightSubLeft_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
            if (U8 == input1_dtype) {
                status |= vsi_nn_kernel_gpu_add_param(node, "input1_ZP", &input1ZP);
                status |= vsi_nn_kernel_gpu_add_param(node, "input1Scale", &input1_scale);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZPtoFp32_part0_4x4", &uniU8SubZPtoFp32_part0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZPtoFp32_part1_4x4", &uniU8SubZPtoFp32_part1_4x4);
            }
            else if (F16 == input1_dtype) {
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniFp16toFp32_part0_4x4", &uniFp16toFp32_part0_4x4);
                status |= vsi_nn_kernel_gpu_add_param(
                    node, "uniFp16toFp32_part1_4x4", &uniFp16toFp32_part1_4x4);
            }
        }
        else if (BF16 == input0_dtype && BF16 == input1_dtype &&
                   BF16 == output_dtype) {
            gpu_dp_inst_t uniBF16toFp32_part0_2x8 = {
                {
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x01050004, 0x03070206, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniBF16toFp32_part1_2x8 = {
                {
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x05050404, 0x07070606, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvBF16toF32_odd_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x02050004, 0x06070406, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvBF16toF32_even_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x03050104, 0x07070506, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniBF16toFp32_part0_2x8", &uniBF16toFp32_part0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniBF16toFp32_part1_2x8", &uniBF16toFp32_part1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniConvBF16toF32_odd_2x8", &uniConvBF16toF32_odd_2x8);
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniConvBF16toF32_even_2x8", &uniConvBF16toF32_even_2x8);
        }
        else if (((I16 == input0_dtype && I16 == input1_dtype &&
                    I16 == output_dtype)) ||
                   ((I8 == input0_dtype && I8 == input1_dtype &&
                     I8 == output_dtype))) {
            float dfpScale = input0_scale * output_scale;
            gpu_dp_inst_t uniDFPtoFp32_part0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniDFPtoFp32_part1_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniDFPtoFp32_left_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00020000, 0x00060004, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniRightSubLeft_4x4 = {{
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00230001, 0x00670045, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            status |= vsi_nn_kernel_gpu_add_param(node, "input1_scale", &input1_scale);
            status |= vsi_nn_kernel_gpu_add_param(node, "dfpScale", &dfpScale);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniDFPtoFp32_part0_4x4", &uniDFPtoFp32_part0_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniDFPtoFp32_part1_4x4", &uniDFPtoFp32_part1_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniDFPtoFp32_left_4x4", &uniDFPtoFp32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniRightSubLeft_4x4", &uniRightSubLeft_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
        }
        else {
            VSILOGE("input or output's format is not support");
            status = VSI_FAILURE;
        }
    }
    CHECK_STATUS_FAIL_GOTO(status, final);

    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;

    gpu_param.dim = 2;
    gpu_param.global_size[0] = gpu_align_p2(
        (out_width + gpu_param.global_scale[0] - 1) /
         gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = ((out_height + gpu_param.global_scale[1] - 1) /
         gpu_param.global_scale[1]);

    status = vsi_nn_kernel_gpu_config(node, &gpu_param);

    final:
#define SAFE_FREE_TENSOR_ATTR(_PTR)               \
    if (_PTR) {                                   \
        vsi_nn_kernel_tensor_attr_release(&_PTR); \
        _PTR = NULL;                              \
    }
        SAFE_FREE_TENSOR_ATTR(output_attr);
        SAFE_FREE_TENSOR_ATTR(input_attr[0]);
        SAFE_FREE_TENSOR_ATTR(input_attr[1]);

        return status;

} /* _bilinear_grid_sample_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype, in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _bilinear_grid_sample_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _bilinear_grid_sample_kernel_map );
    vx_param_description_t * param_def  = _bilinear_grid_sample_kernel_param_def;
    vx_kernel_initialize_f  initializer = _bilinear_grid_sample_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype(inputs[1]->attr.dtype.vx_type);
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = BILINEAR_GRID_SAMPLE_HASH_KEY(in0_dtype, in1_dtype, out_dtype);

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
        kernel->info.numParams   = _cnt_of_array( _bilinear_grid_sample_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_BILINEAR_GRID_SAMPLE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t final_shape[VSI_NN_MAX_DIM_NUM] = {1, 1, 1, 1};
    uint32_t final_in1_rank = 0;
    vsi_nn_tensor_t* rs_tensors = NULL;
    vsi_nn_tensor_t* final_tensors[3] = {NULL};
    vsi_nn_kernel_dtype_e in0_dtype;
    uint32_t pad_val = 0;
    int32_t align_corners =
        vsi_nn_kernel_param_get_int32(params, "align_corners");

    // Check if gpu can support the size
    if (!vsi_nn_kernel_gpu_check_shape(inputs[0]->attr.size,
                                       inputs[0]->attr.dim_num)) {
        return NULL;
    }

    if (!vsi_nn_kernel_gpu_check_shape(inputs[1]->attr.size,
                                       inputs[1]->attr.dim_num)) {
        return NULL;
    }

    final_tensors[0] = inputs[0];

    if (inputs[1]->attr.dim_num >= 3) {

        final_shape[0] = inputs[1]->attr.size[1] * inputs[1]->attr.size[0];
        final_shape[1] = inputs[1]->attr.size[2];
        final_shape[2] = 1;
        final_shape[3] = inputs[1]->attr.dim_num > 3 ? inputs[1]->attr.size[3] : 1;
        final_in1_rank =
            inputs[1]->attr.dim_num == 3 ? 2 : inputs[1]->attr.dim_num;
        if (!vsi_nn_kernel_gpu_check_shape(final_shape, final_in1_rank)) {
            return NULL;
        }

        rs_tensors = vsi_nn_reshape_tensor(graph, inputs[1], final_shape, final_in1_rank);
        final_tensors[1] = rs_tensors;
    } else {
        final_tensors[1] = inputs[1];
    }
    final_tensors[2] = outputs[0];

    in0_dtype = vsi_nn_kernel_map_dtype(inputs[0]->attr.dtype.vx_type);
    if (U8 == in0_dtype) {
        pad_val = inputs[0]->attr.dtype.zero_point;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BILINEAR_GRID_SAMPLE_PARAM_NUM,
                    final_tensors, input_num, &final_tensors[2], output_num );
            node_params[SCALAR_ALIGN_CORNERS] =
                vsi_nn_kernel_scalar_create(graph, I32, &align_corners);
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _BILINEAR_GRID_SAMPLE_PARAM_NUM );
            VSI_ASSERT(status == VSI_SUCCESS);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_ALIGN_CORNERS]);
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U32 = pad_val;
                status = vxSetNodeAttribute(
                    (vx_node)node, VX_NODE_BORDER, &border, sizeof(border));
                CHECK_STATUS(status);
            }
        }
    }

    vsi_safe_release_tensor(rs_tensors);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( bilinear_grid_sample, _setup )


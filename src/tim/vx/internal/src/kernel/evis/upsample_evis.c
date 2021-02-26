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
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_UPSAMPLE,
} _internal_kernel_e;

#define _UPSAMPLE_KERNEL_SOURCE(suffix)      "upsample_"#suffix

// Add kernel hashtable here
#define UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, same_type_flag, _image_2d ) \
        ((IN0_DTYPE << 20) | (IN1_DTYPE << 12) | (OUT_DTYPE << 4) | (same_type_flag << 2) | (_image_2d))

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0, 0 ), \
          CVIVANTE_NAMESPACE("evis.upsample_"#IN0_DTYPE"_"#IN1_DTYPE"to_"#OUT_DTYPE), \
          _UPSAMPLE_KERNEL_SOURCE(IN0_DTYPE) }

#define PACK_KERNEL_MAP_SAME_TYPE( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1, 0 ), \
          CVIVANTE_NAMESPACE("evis.upsample_"#IN0_DTYPE"_"#IN1_DTYPE"to_"#OUT_DTYPE"_SAME"), \
          _UPSAMPLE_KERNEL_SOURCE(IN0_DTYPE) }

#define PACK_KERNEL_MAP_2D( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0, 1 ), \
          CVIVANTE_NAMESPACE("evis.upsample_"#IN0_DTYPE"_"#IN1_DTYPE"to_"#OUT_DTYPE"_2D"), \
          _UPSAMPLE_KERNEL_SOURCE(IN0_DTYPE) }

#define PACK_KERNEL_MAP_SAME_TYPE_2D( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1, 1 ), \
          CVIVANTE_NAMESPACE("evis.upsample_"#IN0_DTYPE"_"#IN1_DTYPE"to_"#OUT_DTYPE"_SAME_2D"), \
          _UPSAMPLE_KERNEL_SOURCE(IN0_DTYPE) }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _upsample_kernel_map[] =
{
    PACK_KERNEL_MAP( I16, U8,  I16),
    PACK_KERNEL_MAP( I16, I16, I16),
    PACK_KERNEL_MAP( I16, U8,  F16),
    PACK_KERNEL_MAP( I16, I16, F16),
    PACK_KERNEL_MAP( I8,  U8,  I8),
    PACK_KERNEL_MAP( I8,  U8,  F16),
    PACK_KERNEL_MAP( U8,  U8,  U8),
    PACK_KERNEL_MAP( U8,  U8,  F16),
    PACK_KERNEL_MAP( F16, U8,  U8),
    PACK_KERNEL_MAP( F16, I16, U8),
    PACK_KERNEL_MAP( F16, U8,  I16),
    PACK_KERNEL_MAP( F16, U8,  I8),
    PACK_KERNEL_MAP_SAME_TYPE( I16, U8, I16),
    PACK_KERNEL_MAP_SAME_TYPE( I8,  U8, I8),
    PACK_KERNEL_MAP_SAME_TYPE( U8,  U8, U8),
    PACK_KERNEL_MAP_2D( I16, U8,  I16),
    PACK_KERNEL_MAP_2D( I16, I16, I16),
    PACK_KERNEL_MAP_2D( I16, U8,  F16),
    PACK_KERNEL_MAP_2D( I16, I16, F16),
    PACK_KERNEL_MAP_2D( I8,  U8,  I8),
    PACK_KERNEL_MAP_2D( I8,  U8,  F16),
    PACK_KERNEL_MAP_2D( U8,  U8,  U8),
    PACK_KERNEL_MAP_2D( U8,  U8,  F16),
    PACK_KERNEL_MAP_2D( F16, U8,  U8),
    PACK_KERNEL_MAP_2D( F16, I16, U8),
    PACK_KERNEL_MAP_2D( F16, U8,  I16),
    PACK_KERNEL_MAP_2D( F16, U8,  I8),
    PACK_KERNEL_MAP_SAME_TYPE_2D( I16, U8, I16),
    PACK_KERNEL_MAP_SAME_TYPE_2D( I8,  U8, I8),
    PACK_KERNEL_MAP_SAME_TYPE_2D( U8,  U8, U8),
};


/*
 * Kernel params
 */
static vx_param_description_t _upsample_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _UPSAMPLE_PARAM_NUM  _cnt_of_array( _upsample_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_upsample_initializer)
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
    vsi_nn_kernel_tensor_attr_t *input_attr    = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t *axis_attr     = NULL;
    vsi_int_array_t * input_shape              = NULL;
    vsi_nn_kernel_dtype_e src_dtype            = F16;
    vsi_nn_kernel_dtype_e dst_dtype            = F16;
    vsi_nn_kernel_dtype_e axis_dtype           = F16;
    int32_t  input_fl                          = 0;
    int32_t  output_fl                         = 0;
    uint16_t M0                                = 0;
    int32_t  postShift                         = 0;
    float    inputScale                        = 1.0f;
    int32_t  input_ZP                          = 0;
    float    outputScale                       = 1.0f;
    int32_t  output_ZP                         = 0;
    float    factorOut                         = 1.0f;
    vsi_bool image_2d                          = FALSE;

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    axis_attr   = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( axis_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );
    input_shape   = input_attr->shape;
    src_dtype     = input_attr->dtype;
    dst_dtype     = output_attr->dtype;
    axis_dtype    = axis_attr->dtype;

    if( input_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        input_fl = input_attr->dfp.fl;
        if (input_fl > 0)
        {
            inputScale = 1.0f / (float) ((int64_t)1 << input_fl);
        }
        else
        {
            inputScale = (float)((int64_t)1 << -input_fl);
        }
    }
    else if( input_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        inputScale  = input_attr->asymm.scale;
        input_ZP    = input_attr->asymm.zero_point;
    }

    if( output_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        output_fl = output_attr->dfp.fl;
        if (output_fl > 0)
        {
            outputScale = 1.0f / (float) ((int64_t)1 << output_fl);
        }
        else
        {
            outputScale = (float)((int64_t)1 << -output_fl);
        }
    }
    else if( output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        outputScale  = output_attr->asymm.scale;
        output_ZP    = output_attr->asymm.zero_point;
    }

    factorOut = 1.0f / outputScale;


    gpu_quantize_multiplier_16bit(inputScale / outputScale, &M0, &postShift);

    image_2d = (vsi_bool)(input_shape->size < 3 || 1 == input_shape->data[2]);

    if (BF16 == src_dtype && BF16 == dst_dtype)
    {
        src_dtype = F16;
        dst_dtype = F16;
    }

    if ((F16 == src_dtype) && (F16 == dst_dtype))
    {
        src_dtype  = I16;
        dst_dtype  = I16;
    }

    if (I8 == axis_dtype)
    {
        axis_dtype = U8;
    }

    if (I8 == src_dtype || U8 == src_dtype
       || (F16 == src_dtype && U8  == dst_dtype && U8  == axis_dtype))
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }
    else
    {
        gpu_param.global_scale[0]  = 4;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }

    gpu_param.dim = image_2d ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (input_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (input_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = image_2d ? 1 : (
            (input_shape->data[2] + gpu_param.global_scale[2] - 1)
            / gpu_param.global_scale[2]);

    if((I8  == src_dtype && I8  == dst_dtype)
    || (U8  == src_dtype && U8  == dst_dtype)
    || (U8  == src_dtype && F16 == dst_dtype)
    || (F16 == src_dtype && U8  == dst_dtype && U8 == axis_dtype)
        )
    {
        vx_bool is_same_quant_type = (vx_bool)((input_attr->dfp.fl == output_attr->dfp.fl
                                        && input_attr->quant == VSI_NN_KERNEL_QUANT_DFP
                                        && output_attr->quant == VSI_NN_KERNEL_QUANT_DFP)
                                        || ((input_attr->asymm.zero_point == output_attr->asymm.zero_point)
                                        && (input_attr->asymm.scale == output_attr->asymm.scale)
                                        && input_attr->quant  == VSI_NN_KERNEL_QUANT_ASYMM
                                        && output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM));

        if (is_same_quant_type)
        {
            status = vsi_nn_kernel_gpu_add_param(node, "input_ZP", &input_ZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if ((I8  == src_dtype && I8  == dst_dtype)
               || (U8  == src_dtype && U8  == dst_dtype))
        {
            gpu_dp_inst_t uniU8SubZP_MulM_2x8 = {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8SubZP_MulM_Hi_2x8 = {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniS16AddOutZP_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            }, GPU_DP_TYPE_16};
            vx_uint32 uniS16MoveValue_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            uint32_t idx                   = 0;
            uint32_t packed_outputZP[4]    = {0};

            for (idx = 0; idx < 4; idx ++)
            {
                vx_uint16  zp = (vx_uint16)(output_ZP & 0xFFFF);
                packed_outputZP[idx] = (zp << 16) | zp;
            }

            uniU8SubZP_MulM_2x8.data[7]    |= postShift;
            uniU8SubZP_MulM_Hi_2x8.data[7] |= postShift;

            for (idx = 8; idx < 16; idx ++)
            {
                uniU8SubZP_MulM_2x8.data[idx]    = (uint32_t)((M0 << 16) | M0);
                uniU8SubZP_MulM_Hi_2x8.data[idx] = (uint32_t)((M0 << 16) | M0);
            }

            status  = vsi_nn_kernel_gpu_add_param(node, "uniU8SubZP_MulM_2x8", &uniU8SubZP_MulM_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZP_MulM_Hi_2x8", &uniU8SubZP_MulM_Hi_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniS16AddOutZP_2x8", &uniS16AddOutZP_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniS16MoveValue_2x8", &uniS16MoveValue_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "packed_outputZP", packed_outputZP);
            status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP", &input_ZP);
            CHECK_STATUS_FAIL_GOTO(status, final );

        }
        else if (!is_same_quant_type)
        {
            // uniforms
            gpu_dp_inst_t uniConvertDirUint8Fp32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertEndUint8Fp32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertTrdUint8Fp32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00090008, 0x000b000a, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertFthUint8Fp32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x000d000c, 0x000f000e, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniMulMinusZpUint8_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniMulMinusZp2Uint8_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniMulMinusZp3Uint8_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00090008, 0x000b000a, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniMulMinusZp4Uint8_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x000d000c, 0x000f000e, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniF16MulMultipiler_PostShft_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniS16AddOutZP_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            }, GPU_DP_TYPE_16};

            if(F16 == src_dtype && U8  == dst_dtype)
            {
                uint32_t idx                   = 0;
                uint32_t packed_outputZP[4]    = {0};

                for (idx = 0; idx < 4; idx ++)
                {
                    vx_uint8  zp = (vx_uint8)(output_ZP & 0xFF);
                    packed_outputZP[idx] = (zp << 24) | (zp << 16) | (zp << 8) | zp;
                }

                uniF16MulMultipiler_PostShft_2x8.data[7] |= postShift;

                for (idx = 8; idx < 16; idx ++)
                {
                    uniF16MulMultipiler_PostShft_2x8.data[idx] = (uint32_t)(M0);
                }

                status  = vsi_nn_kernel_gpu_add_param(node, "uniF16MulMultipiler_PostShft_2x8",
                                                      &uniF16MulMultipiler_PostShft_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniS16AddOutZP_2x8",
                                                      &uniS16AddOutZP_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "packed_outputZP", packed_outputZP);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            else if(U8 == src_dtype && F16  == dst_dtype)
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniConvertDirUint8Fp32_4x4",
                                                      &uniConvertDirUint8Fp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertEndUint8Fp32_4x4",
                                                      &uniConvertEndUint8Fp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertTrdUint8Fp32_4x4",
                                                      &uniConvertTrdUint8Fp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertFthUint8Fp32_4x4",
                                                      &uniConvertFthUint8Fp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toInt16_2x8",
                                                      &uniConvertInt32toUint8_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "scaleU8Fp16", &inputScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "zpU8Fp16", &input_ZP);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniMulMinusZpUint8_4x4", &uniMulMinusZpUint8_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniMulMinusZp2Uint8_4x4", &uniMulMinusZp2Uint8_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniMulMinusZp3Uint8_4x4", &uniMulMinusZp3Uint8_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniMulMinusZp4Uint8_4x4", &uniMulMinusZp4Uint8_4x4);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
        }
    }
    else if((I16 == src_dtype && I16 == dst_dtype)
            || (I16 == src_dtype && F16 == dst_dtype))
    {
        // uniforms
        gpu_dp_inst_t uniConvertDirInt16Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniQuantInOutInt16_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t ucharMulShort_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertU8toI16_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        if(I16 == src_dtype && I16 == dst_dtype )
        {
            status = vsi_nn_kernel_gpu_add_param(node, "ucharMulShort_2x8",
                                                    &ucharMulShort_2x8);
            if (input_fl != output_fl || I16 == axis_dtype)
            {
                if(input_fl > output_fl)
                {
                    uniQuantInOutInt16_2x8.data[7] = uniQuantInOutInt16_2x8.data[7] | (input_fl - output_fl);
                }
                else
                {
                    uint32_t multiply       = ((int64_t)1 << (output_fl - input_fl));
                    uint32_t i              = 0;

                    for (i = 8; i < 16; i++)
                    {
                        uniQuantInOutInt16_2x8.data[i] = multiply;
                    }
                }

                status |= vsi_nn_kernel_gpu_add_param(node, "uniQuantInOutInt16_2x8",
                                                     &uniQuantInOutInt16_2x8);
            }
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if(I16 == src_dtype && F16 == dst_dtype)
        {
            status  = vsi_nn_kernel_gpu_add_param(node, "uniConvertDirInt16Fp32_4x4",
                                                  &uniConvertDirInt16Fp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertU8toI16_2x8",
                                                  &uniConvertU8toI16_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "inScaleInt16", &inputScale);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
    }
    else if( F16 == src_dtype
        && ( (U8 == dst_dtype && I16 == axis_dtype) || I8 == dst_dtype || I16 == dst_dtype ))
    {
        // uniforms
        gpu_dp_inst_t shortMulShort_8x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t ucharMulShort_8x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertFstFp16Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertSecFp16Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8",
                                              &uniConvertInt32toUint8_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertFstFp16Fp32_4x4",
                                              &uniConvertFstFp16Fp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecFp16Fp32_4x4",
                                              &uniConvertSecFp16Fp32_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
        if(U8 == dst_dtype && I16 == axis_dtype)
        {
            status  = vsi_nn_kernel_gpu_add_param(node, "upOutput_Scale", &outputScale);
            status |= vsi_nn_kernel_gpu_add_param(node, "shortMulShort_8x8", &shortMulShort_8x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "upOutput_ZP", &output_ZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if(I8 == dst_dtype)
        {
            float scale_out_f = 1.0f / outputScale;
            float output_ZP_f = (float)output_ZP;
            status  = vsi_nn_kernel_gpu_add_param(node, "scaleOut", &scale_out_f);
            status |= vsi_nn_kernel_gpu_add_param(node, "outputZp", &output_ZP_f);
            status |= vsi_nn_kernel_gpu_add_param(node, "ucharMulShort_8x8_2", &ucharMulShort_8x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if(I16 == dst_dtype)
        {
            status  = vsi_nn_kernel_gpu_add_param(node, "up_outFlScale_i16", &factorOut);
            status |= vsi_nn_kernel_gpu_add_param(node, "ucharMulShort_8x8_2", &ucharMulShort_8x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
    }
    else if( I8  == src_dtype && F16  == dst_dtype )
    {
        gpu_dp_inst_t uniConvertDirUint8Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertEndUint8Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertTrdUint8Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertFthUint8Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        float inputTail = (float)input_ZP * inputScale * (-1.0f);

        status  = vsi_nn_kernel_gpu_add_param(node, "uniConvertDirUint8Fp32_4x4_2", &uniConvertDirUint8Fp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertEndUint8Fp32_4x4_2", &uniConvertEndUint8Fp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertTrdUint8Fp32_4x4_2", &uniConvertTrdUint8Fp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertFthUint8Fp32_4x4_2", &uniConvertFthUint8Fp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8_2", &uniConvertInt32toUint8_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "scaleIn", &inputScale);
        status |= vsi_nn_kernel_gpu_add_param(node, "inputTail", &inputTail);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );
    if (axis_attr) vsi_nn_kernel_tensor_attr_release( &axis_attr );
    return status;

} /* _upsample_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _upsample_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _upsample_kernel_map );
    vx_param_description_t * param_def  = _upsample_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _upsample_kernel_param_def );
    vx_kernel_initialize_f  initializer = _upsample_initializer;
    uint32_t key;
    uint32_t i;
    vsi_bool is_same_type = FALSE;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if ((BF16 == in0_dtype) && (BF16 == out_dtype))
    {
        in0_dtype  = F16;
        out_dtype  = F16;
    }

    if (I8 == in1_dtype)
    {
        in1_dtype = U8;
    }

    if (((I8 == in0_dtype) && (I8 == out_dtype))
       || ((I16 == in0_dtype) && (I16 == out_dtype))
       || ((U8 == in0_dtype) && (U8 == out_dtype)))
    {
         if ((inputs[0]->attr.dtype.fl == outputs[0]->attr.dtype.fl
            && inputs[0]->attr.dtype.qnt_type  == VSI_NN_QNT_TYPE_DFP
            && outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
            || ((inputs[0]->attr.dtype.zero_point == outputs[0]->attr.dtype.zero_point)
            && (inputs[0]->attr.dtype.scale == outputs[0]->attr.dtype.scale)
            && inputs[0]->attr.dtype.qnt_type  == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC
            && outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC))
            {
                is_same_type = TRUE;
            }
    }

    if ((F16 == in0_dtype) && (F16 == out_dtype))
    {
        in0_dtype  = I16;
        out_dtype  = I16;
        is_same_type = TRUE;
    }

    key = UPSAMPLE_HASH_KEY( in0_dtype, in1_dtype, out_dtype, is_same_type, image_2d );

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }


    if( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
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
    vsi_nn_kernel_node_param_t node_params[_UPSAMPLE_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t  scale_x  = 0;
    int32_t  scale_y  = 0;
    vsi_bool image_2d = FALSE;

    scale_x  = vsi_nn_kernel_param_get_int32(params, "scale_x");
    scale_y  = vsi_nn_kernel_param_get_int32(params, "scale_y");

    if (2 != scale_x || 2 != scale_y)
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( (int32_t*)inputs[1]->attr.size,
                inputs[1]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ))
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _UPSAMPLE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _UPSAMPLE_PARAM_NUM );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( upsample, _setup )


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

#define _POOLWITHARGMAX_KERNEL_SOURCE(suffix)      "poolwithargmax_"#suffix

// Add kernel hashtable here
#define POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, same_type_flag, _image_2d ) \
        ((IN_DTYPE << 20) | (OUT0_DTYPE << 12) | (OUT1_DTYPE << 4) | (same_type_flag << 2) | (_image_2d))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE ) \
        { POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, 0, 0 ), \
          CVIVANTE_NAMESPACE("evis.poolwithargmax_"#IN_DTYPE"to_"#OUT0_DTYPE"_"#OUT1_DTYPE), \
          _POOLWITHARGMAX_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_SAME_TYPE( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE ) \
        { POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, 1, 0 ), \
          CVIVANTE_NAMESPACE("evis.poolwithargmax_"#IN_DTYPE"to_"#OUT0_DTYPE"_"#OUT1_DTYPE"_SAME"), \
          _POOLWITHARGMAX_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE ) \
        { POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, 0, 1 ), \
          CVIVANTE_NAMESPACE("evis.poolwithargmax_"#IN_DTYPE"to_"#OUT0_DTYPE"_"#OUT1_DTYPE"_2D"), \
          _POOLWITHARGMAX_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_SAME_TYPE_2D( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE ) \
        { POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, 1, 1 ), \
          CVIVANTE_NAMESPACE("evis.poolwithargmax_"#IN_DTYPE"to_"#OUT0_DTYPE"_"#OUT1_DTYPE"_SAME_2D"), \
          _POOLWITHARGMAX_KERNEL_SOURCE(IN_DTYPE) }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _poolwithargmax_kernel_map[] =
{
    PACK_KERNEL_MAP( F16, F16, U8 ),
    PACK_KERNEL_MAP( F16, I16, U8 ),
    PACK_KERNEL_MAP( U8,  U8,  U8 ),
    PACK_KERNEL_MAP( U8,  F16, U8 ),
    PACK_KERNEL_MAP( U8,  F16, I16 ),
    PACK_KERNEL_MAP( I8,  I8,  U8 ),
    PACK_KERNEL_MAP( I8,  F16, U8 ),
    PACK_KERNEL_MAP( I16, I16, U8 ),
    PACK_KERNEL_MAP( I16, I16, I16 ),
    PACK_KERNEL_MAP( I16, F16, U8 ),
    PACK_KERNEL_MAP_SAME_TYPE( I8,  I8,  U8 ),
    PACK_KERNEL_MAP_SAME_TYPE( I16, I16, U8 ),
    PACK_KERNEL_MAP_2D( F16, F16, U8 ),
    PACK_KERNEL_MAP_2D( F16, I16, U8 ),
    PACK_KERNEL_MAP_2D( U8,  U8,  U8 ),
    PACK_KERNEL_MAP_2D( U8,  F16, U8 ),
    PACK_KERNEL_MAP_2D( U8,  F16, I16 ),
    PACK_KERNEL_MAP_2D( I8,  I8,  U8 ),
    PACK_KERNEL_MAP_2D( I8,  F16, U8 ),
    PACK_KERNEL_MAP_2D( I16, I16, U8 ),
    PACK_KERNEL_MAP_2D( I16, I16, I16 ),
    PACK_KERNEL_MAP_2D( I16, F16, U8 ),
    PACK_KERNEL_MAP_SAME_TYPE_2D( I8,  I8,  U8 ),
    PACK_KERNEL_MAP_SAME_TYPE_2D( I16, I16, U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _poolwithargmax_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

#define _POOLWITHARGMAX_PARAM_NUM  _cnt_of_array( _poolwithargmax_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_poolwithargmax_initializer)
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
    vsi_size_array_t * input_shape              = NULL;
    vsi_nn_kernel_dtype_e src_dtype            = F16;
    vsi_nn_kernel_dtype_e dst_dtype            = F16;
    int32_t  input_fl                          = 0;
    int32_t  output_fl                         = 0;
    uint16_t M0                                = 0;
    int32_t  postShift                         = 0;
    float    inputScale                        = 1.0f;
    int32_t  input_ZP                          = 0;
    float    outputScale                       = 1.0f;
    int32_t  output_ZP                         = 0;
    vsi_bool image_2d                          = FALSE;

    VSI_UNREFERENCED(param_size);

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );
    input_shape   = input_attr->shape;
    src_dtype     = input_attr->dtype;
    dst_dtype     = output_attr->dtype;
    inputScale    = input_attr->scale;
    input_ZP      = input_attr->zero_point;
    outputScale   = output_attr->scale;
    output_ZP     = output_attr->zero_point;

    if( input_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        input_fl = input_attr->dfp.fl;
    }

    if( output_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        output_fl = output_attr->dfp.fl;
    }

    if ( ( input_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
       && ( output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM ) )
    {
        gpu_quantize_multiplier_16bit(inputScale / outputScale, &M0, &postShift);
    }

    image_2d = (vsi_bool)(input_shape->size < 3 || 1 == input_shape->data[2]);


    if (BF16 == src_dtype && BF16 == dst_dtype)
    {
        src_dtype = F16;
        dst_dtype = F16;
    }


    if (I8 == src_dtype || U8 == src_dtype)
    {
        gpu_param.global_scale[0]  = 16;
        gpu_param.global_scale[1]  = 2;
        gpu_param.global_scale[2]  = 1;
    }
    else
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 2;
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


    if(I8 == src_dtype || U8 == src_dtype)
    {
        gpu_dp_inst_t poolingEncodeInt8_0 = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x32321010, 0x76765454, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000700, // AccumType, ConstantType, and PostShift
            0x00400080, 0x00100020, 0x00400080, 0x00100020,
            0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t poolingEncodeInt8_1 = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0xbaba9898, 0xfefedcdc, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000700, // AccumType, ConstantType, and PostShift
            0x00400080, 0x00100020, 0x00400080, 0x00100020,
            0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8EvenBinSubZP_MulM_2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x06040200, 0x0e0c0a08, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00020001, 0x00020001, 0x00020001, 0x00020001,
            0x00020001, 0x00020001, 0x00020001, 0x00020001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniEncodeUint8_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x8628c020, 0x6ad0a49c, 0xe128bd8e, 0xacde96ac, 0xff9eeef1, // BinSelect
            0x00000700, // AccumType, ConstantType, and PostShift
            0x10204080, 0x10204080, 0x10204080, 0x10204080,
            0x10204080, 0x10204080, 0x10204080, 0x10204080 // Constant
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

        if ((U8 == src_dtype && U8 == dst_dtype)
            || (I8 == src_dtype && I8 == dst_dtype))
        {
            vx_uint32 idx                   = 0;
            vx_uint32 packed_outputZP[4]    = {0};

            for (idx = 0; idx < 4; idx ++)
            {
                vx_uint8  zp = (vx_uint8)(output_ZP & 0xFF);
                packed_outputZP[idx] = (zp << 24) | (zp << 16) | (zp << 8) | zp;
            }

            uniU8EvenBinSubZP_MulM_2x8.data[7] |= postShift;

            for (idx = 8; idx < 16; idx ++)
            {
                uniU8EvenBinSubZP_MulM_2x8.data[idx] = (vx_uint32)((M0 << 16) | M0);
            }

            status  = vsi_nn_kernel_gpu_add_param(node, "uniU8EvenBinSubZP_MulM_2x8",
                                                   &uniU8EvenBinSubZP_MulM_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniS16AddOutZP_2x8",
                                                   &uniS16AddOutZP_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "packed_outputZP", packed_outputZP);
            status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP", &input_ZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        if(U8 == src_dtype)
        {
            status = vsi_nn_kernel_gpu_add_param(node, "uniEncodeUint8_4x8",
                                            &uniEncodeUint8_4x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else
        {
            status  = vsi_nn_kernel_gpu_add_param(node, "poolingEncodeInt8_0",
                                            &poolingEncodeInt8_0);
            status |= vsi_nn_kernel_gpu_add_param(node, "poolingEncodeInt8_1",
                                            &poolingEncodeInt8_1);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        if(F16 == dst_dtype)
        {
            gpu_dp_inst_t uniConvertUint8ToFp32_4x4 = {{
                0x09090909, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertSubZpUint8Fp32_4x4 = {{
                0x09090905, // TCfg
                0x04040404, // ASelt
                0x00050004, 0x00070006, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniPackHalf8_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertEvenU8ToFp32_4x4 = {{
                0x09090905, // TCfg
                0x04040404, // ASelt
                0x00020000, 0x00060004, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0xffff0001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertEvenU8SubZpToFp32_4x4 = {{
                0x09090909, // TCfg
                0x04040404, // ASelt
                0x000a0008, 0x000e000c, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param(node, "uniPackHalf8_2x8",
                                                 &uniPackHalf8_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertEvenU8ToFp32_4x4",
                                                 &uniConvertEvenU8ToFp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertEvenU8SubZpToFp32_4x4",
                                                 &uniConvertEvenU8SubZpToFp32_4x4);
            if(U8 == src_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertUint8ToFp32_4x4",
                                                &uniConvertUint8ToFp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSubZpUint8Fp32_4x4",
                                                &uniConvertSubZpUint8Fp32_4x4);
            }
            status |= vsi_nn_kernel_gpu_add_param(node, "inputScale", &inputScale);
            status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP", &input_ZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
    }
    else
    {
        gpu_dp_inst_t poolingEncode = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x32321010, 0x76765454, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000700, // AccumType, ConstantType, and PostShift
            0x00400080, 0x00100020, 0x00400080, 0x00100020,
            0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
        }, GPU_DP_TYPE_16};

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
        gpu_dp_inst_t uniConvertEndInt16Fp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniPackHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniQuantInOutInt16Even_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        if(F16 == src_dtype)
        {
            status = vsi_nn_kernel_gpu_add_param(node, "poolingEncode", &poolingEncode);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if(I16 == src_dtype)
        {
            if(F16 == dst_dtype)
            {
                status  = vsi_nn_kernel_gpu_add_param(node, "uniPackHalf8_2x8_2",
                            &uniPackHalf8_2x8);
                status |= vsi_nn_kernel_gpu_add_param(node, "input_fl_scale_i16", &inputScale);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertDirInt16Fp32_4x4",
                                                    &uniConvertDirInt16Fp32_4x4);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertEndInt16Fp32_4x4",
                                                    &uniConvertEndInt16Fp32_4x4);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            status = vsi_nn_kernel_gpu_add_param(node, "poolingEncode2", &poolingEncode);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        if (I16 == dst_dtype)
        {
            if(input_fl > output_fl)
            {
                uniQuantInOutInt16Even_4x4.data[7] = uniQuantInOutInt16Even_4x4.data[7] | (input_fl - output_fl);
            }
            else
            {
                vx_uint32 multiply       = ((int64_t)1 << (output_fl - input_fl));
                vx_uint32 i              = 0;

                for (i = 8; i < 16; i+=2)
                {
                    uniQuantInOutInt16Even_4x4.data[i] = multiply;
                }
            }

            status  = vsi_nn_kernel_gpu_add_param(node, "uniQuantInOutInt16Even_4x4", &uniQuantInOutInt16Even_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );

    return status;

} /* _poolwithargmax_initializer() */



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
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out0_dtype;
    vsi_nn_kernel_dtype_e out1_dtype;
    const _kernel_map_type * kernel_map = _poolwithargmax_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _poolwithargmax_kernel_map );
    vx_param_description_t * param_def  = _poolwithargmax_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _poolwithargmax_kernel_param_def );
    vx_kernel_initialize_f  initializer = _poolwithargmax_initializer;
    uint32_t key;
    uint32_t i;
    vsi_bool is_same_type = FALSE;

    in_dtype   = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out0_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    out1_dtype = vsi_nn_kernel_map_dtype( outputs[1]->attr.dtype.vx_type );

    if ((BF16 == in_dtype) && (BF16 == out0_dtype))
    {
        in_dtype   = F16;
        out0_dtype = F16;
    }

    if (I8 == out1_dtype)
    {
        out1_dtype = U8;
    }

    if (((I8 == in_dtype) && (I8 == out0_dtype)) || ((I16 == in_dtype) && (I16 == out0_dtype)))
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

    key = POOLWITHARGMAX_HASH_KEY( in_dtype, out0_dtype, out1_dtype, is_same_type, image_2d );

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
    vsi_nn_kernel_node_param_t node_params[_POOLWITHARGMAX_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t  ksize_x  = 0;
    int32_t  ksize_y  = 0;
    int32_t  stride_x = 0;
    int32_t  stride_y = 0;
    int32_t  pad_x    = 0;
    int32_t  pad_y    = 0;
    vsi_bool image_2d = FALSE;

    ksize_x  = vsi_nn_kernel_param_get_int32(params, "ksize_x");
    ksize_y  = vsi_nn_kernel_param_get_int32(params, "ksize_y");
    stride_x = vsi_nn_kernel_param_get_int32(params, "stride_x");
    stride_y = vsi_nn_kernel_param_get_int32(params, "stride_y");
    pad_x    = vsi_nn_kernel_param_get_int32(params, "pad_x");
    pad_y    = vsi_nn_kernel_param_get_int32(params, "pad_y");

    if ((2 != ksize_x) || (2 != ksize_y) || (2 != stride_x) || (2 != stride_y) || (0 != pad_x) || (0 != pad_y))
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[1]->attr.size,
                outputs[1]->attr.dim_num ))
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _POOLWITHARGMAX_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _POOLWITHARGMAX_PARAM_NUM );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( poolwithargmax, _setup )


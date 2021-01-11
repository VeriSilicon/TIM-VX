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

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _CPU_IO_NUM         (_INPUT_NUM + _OUTPUT_NUM)
#define _ADD_MEAN_STD_NORM_KERNEL_SOURCE      "add_mean_std_norm"

// Add kernel hashtable here
#define ADD_MEAN_STD_NORM_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        ((IN0_DTYPE << 20) | ( IN1_DTYPE << 12 ) | ( OUT_DTYPE << 4) )

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { ADD_MEAN_STD_NORM_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), \
        CVIVANTE_NAMESPACE("evis.add_mean_std_norm_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE), \
         _ADD_MEAN_STD_NORM_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _add_mean_std_norm_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F16, F16, F16 ),
    PACK_KERNEL_MAP( U8 , U8 , F16 ),
    PACK_KERNEL_MAP( I16, I16, F16 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _add_mean_std_norm_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _ADD_MEAN_STD_NORM_PARAM_NUM  _cnt_of_array( _add_mean_std_norm_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_add_mean_std_norm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VX_FAILURE;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vx_tensor              input0        = (vx_tensor)param[0];
    vx_tensor              input1        = (vx_tensor)param[1];
    vx_tensor              output        = (vx_tensor)param[2];
    vsi_nn_kernel_dtype_e  input_dtype   = F16;
    vsi_nn_kernel_dtype_e  output_dtype  = F16;
    vsi_nn_kernel_tensor_attr_t *input0_attr = NULL, *input1_attr = NULL, *output_attr = NULL;
    vsi_int_array_t             *input_shape  = NULL;
    float   scaleIn     = 1.0f;
    int32_t input_ZP    = 0;
    float   scaleIn1    = 1.0f;
    int32_t input_ZP1   = 0;
    float   scaleOut    = 1.0f;
    int32_t output_ZP   = 0;
    int32_t fixpoint = 0, fixpoint1 = 0, fixpoint_out = 0;
    float   inScale_dfp, inScale_dfp1;
    float   eps        = 0.0f;
    float   rsEps      = 0.0f;
    float   dimRatio   = 0.0f;
    int32_t width      = 0;

    input0_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input0);
    CHECK_PTR_FAIL_GOTO( input0_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    input1_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input1);
    CHECK_PTR_FAIL_GOTO( input1_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    input_shape   = input0_attr->shape;
    input_dtype   = input0_attr->dtype;
    output_dtype  = output_attr->dtype;
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[_CPU_IO_NUM], &(eps));
    rsEps    = (float)(1.0f / sqrtf(eps));
    dimRatio = (float)(1.0 / (input_shape->data[0]));


    if ( VSI_NN_KERNEL_QUANT_DFP == input0_attr->quant )
    {
        fixpoint   = input0_attr->dfp.fl;
    }
    else if ( VSI_NN_KERNEL_QUANT_ASYMM == input0_attr->quant )
    {
        input_ZP   = input0_attr->asymm.zero_point;
        scaleIn    = input0_attr->asymm.scale;
    }
    else
    {
        input_ZP   = 0;
        scaleIn    = 1.0f;
    }

    //input1
    if ( VSI_NN_KERNEL_QUANT_DFP == input1_attr->quant )
    {
        fixpoint1  = input1_attr->dfp.fl;
    }
    else if ( VSI_NN_KERNEL_QUANT_ASYMM == input1_attr->quant )
    {
        input_ZP1  = input1_attr->asymm.zero_point;
        scaleIn1   = input1_attr->asymm.scale;
    }
    else
    {
        input_ZP1   = 0;
        scaleIn1    = 1.0f;
    }

    //output
    if ( VSI_NN_KERNEL_QUANT_DFP == output_attr->quant )
    {
        fixpoint_out = output_attr->dfp.fl;
        if (fixpoint_out >= 0)
        {
            scaleOut = 1.0f / (vx_float32) ((int64_t)1 << fixpoint_out);
        }
        else
        {
            scaleOut = (vx_float32) ((int64_t)1 << -fixpoint_out);
        }
        output_ZP = 0;
    }
    else if ( VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant )
    {
        output_ZP  = output_attr->asymm.zero_point;
        scaleOut   = output_attr->asymm.scale;
    }
    else
    {
        output_ZP   = 0;
        scaleOut    = 1.0f;
    }

    if (fixpoint >= 0)
    {
        inScale_dfp = 1.0f / (vx_float32) ((int64_t)1 << fixpoint);
    }
    else
    {
        inScale_dfp = (vx_float32) ((int64_t)1 << -fixpoint);
    }

    if (fixpoint1 >= 0)
    {
        inScale_dfp1 = 1.0f / (vx_float32) ((int64_t)1 << fixpoint1);
    }
    else
    {
        inScale_dfp1 = (vx_float32) ((int64_t)1 << -fixpoint1);
    }

    gpu_param.global_offset[0] = 0;
    gpu_param.global_offset[1] = 0;
    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.local_size[0]    = 16;
    gpu_param.local_size[1]    = 1;
    gpu_param.global_size[0]   = 16;
    gpu_param.global_size[1]   = gpu_align_p2( (input_shape->data[1] + gpu_param.global_scale[1] - 1)
                                             / gpu_param.global_scale[1], gpu_param.local_size[1] );

    {
        gpu_dp_inst_t uniAddFp16_2x8 = {{
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x33221100, 0x77665544, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniFp16SumSqr_dp8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniAddFp16toFp32Lo_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniAddFp16toFp32Hi_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00550044, 0x00770066, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
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
        status  = vsi_nn_kernel_gpu_add_param(node, "uniAddFp16_2x8", &uniAddFp16_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniFp16SumSqr_dp8x2", &uniFp16SumSqr_dp8x2);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniAddFp16toFp32Lo_4x4", &uniAddFp16toFp32Lo_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniAddFp16toFp32Hi_4x4", &uniAddFp16toFp32Hi_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if( U8 == input_dtype && F16 == output_dtype )
    {
        vx_uint16  M0                 = 0;
        vx_int8    postShift          = 0;
        vx_uint32  multAndoutZP0[2]   = {0};
        vx_uint32  multAndoutZP1[2]   = {0};

        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8MulAndPostShift_1_Lo_2x8 = {{
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        vsi_nn_GetFP32MultiAndPostShift(scaleIn / scaleOut, &M0, &postShift);
        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((output_ZP << postShift) - input_ZP * M0);
        uniU8MulAndPostShift_0_Lo_2x8.data[7] |= (postShift & 0x1F);

        vsi_nn_GetFP32MultiAndPostShift(scaleIn1 / scaleOut, &M0, &postShift);
        multAndoutZP1[0] = (vx_uint32)(M0);
        multAndoutZP1[1] = (vx_uint32)((output_ZP << postShift) - input_ZP1 * M0);
        uniU8MulAndPostShift_1_Lo_2x8.data[7] |= (postShift & 0x1F);

        status  = vsi_nn_kernel_gpu_add_param( node, "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
        status |= vsi_nn_kernel_gpu_add_param( node, "uniU8MulAndPostShift_1_Lo_2x8", &uniU8MulAndPostShift_1_Lo_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if( I16 == input_dtype && F16 == output_dtype )
    {
        gpu_dp_inst_t uniConvertInt16ScaleToFp32Fst_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertInt16ScaleToFp32Sec_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        status  = vsi_nn_kernel_gpu_add_param(node, "uniConvertInt16ScaleToFp32Fst_4x4",
                       &uniConvertInt16ScaleToFp32Fst_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertInt16ScaleToFp32Sec_4x4",
                       &uniConvertInt16ScaleToFp32Sec_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "inScale_i16", &inScale_dfp);
        status |= vsi_nn_kernel_gpu_add_param(node, "inScale1_i16", &inScale_dfp1);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    width   = input_shape->data[0];
    status  = vsi_nn_kernel_gpu_add_param(node, "width", &width);
    status |= vsi_nn_kernel_gpu_add_param(node, "dimRatio", &dimRatio);
    status |= vsi_nn_kernel_gpu_add_param(node, "rsEps", &rsEps);
    CHECK_STATUS_FAIL_GOTO(status, final );


    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (input0_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input0_attr);
    }
    if (input1_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input1_attr);
    }
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }
    return status;

} /* _add_mean_std_norm_initializer() */



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
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _add_mean_std_norm_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _add_mean_std_norm_kernel_map );
    vx_param_description_t * param_def  = _add_mean_std_norm_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _add_mean_std_norm_kernel_param_def );
    vx_kernel_initialize_f  initializer = _add_mean_std_norm_initializer;
    uint32_t key;
    uint32_t i;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = ADD_MEAN_STD_NORM_HASH_KEY( in0_dtype, in1_dtype, out_dtype );

    for( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if( i < (uint32_t)kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_ADD_MEAN_STD_NORM_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" );

    status = _query_kernel( kernel, inputs, outputs );

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U32 = 0;
            border.constant_value.S16 = 0;
            border.constant_value.U8 = 0;
            if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
            {
                border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
            }
            status  = vsi_nn_kernel_node_set_border( node, &border );
            VSI_ASSERT( status == VSI_SUCCESS );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ADD_MEAN_STD_NORM_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[_CPU_IO_NUM] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ADD_MEAN_STD_NORM_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM] );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( add_mean_std_norm, _setup )


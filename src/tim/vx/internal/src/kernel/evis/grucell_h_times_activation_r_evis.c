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
typedef enum _grucell_nn_activation_type_e
{
    SIGMOID = VSI_NN_ACT_SIGMOID,
    HSIGMOID = VSI_NN_ACT_HARD_SIGMOID,
}grucell_nn_activation_type_e;

#define _GRUCELL_H_TIMES_ACTIVATION_R_KERNEL_SOURCE      "grucell_h_times_activation_r"

// Add kernel hashtable here
#define GRUCELL_H_TIMES_ACTIVATION_R_HASH_KEY( HSTATE_DTYPE, IN_FC_DTYPE, OUT_TYPE, REC_ACT ) \
        (( HSTATE_DTYPE ) | ( IN_FC_DTYPE << 6 ) | ( OUT_TYPE << 12 ) | ( REC_ACT << 18 ))
#define PACK_KERNEL_MAP( HSTATE_DTYPE, IN_FC_DTYPE, OUT_TYPE, REC_ACT ) \
        { GRUCELL_H_TIMES_ACTIVATION_R_HASH_KEY( HSTATE_DTYPE, IN_FC_DTYPE, OUT_TYPE, REC_ACT ), \
CVIVANTE_NAMESPACE("evis.grucell_h_times_activation_r_"#HSTATE_DTYPE"_"#IN_FC_DTYPE"to"#OUT_TYPE"_"#REC_ACT), \
_GRUCELL_H_TIMES_ACTIVATION_R_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _grucell_h_times_activation_r_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8,   F16,  F16,  SIGMOID ),
    PACK_KERNEL_MAP( I8,   F16,  F16,  SIGMOID ),
    PACK_KERNEL_MAP( I16,  F16,  F16,  SIGMOID ),
    PACK_KERNEL_MAP( F16,  F16,  F16,  SIGMOID ),
    PACK_KERNEL_MAP( BF16, BF16, BF16, SIGMOID ),
    PACK_KERNEL_MAP( U8,   F16,  F16,  HSIGMOID ),
    PACK_KERNEL_MAP( I8,   F16,  F16,  HSIGMOID ),
    PACK_KERNEL_MAP( I16,  F16,  F16,  HSIGMOID ),
    PACK_KERNEL_MAP( F16,  F16,  F16,  HSIGMOID ),
    PACK_KERNEL_MAP( BF16, BF16, BF16, HSIGMOID ),
};

/*
 * Kernel params
 */
static vx_param_description_t _grucell_h_times_activation_r_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM  _cnt_of_array( _grucell_h_times_activation_r_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_grucell_h_times_activation_r_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_status                   status                 = VSI_FAILURE;
    vsi_nn_kernel_tensor_t       output                 = NULL;
    float                        hstate_in_scale        = 1.0f;
    float                        hstate_in_tail         = 0;
    uint32_t                     i                      = 0;
    uint32_t                     pack_key               = 0;
    vsi_nn_kernel_tensor_attr_t* input_attr[2]          = {NULL};
    vsi_nn_kernel_tensor_attr_t* output_attr[1]         = {NULL};
#define _PACK_SELECT_KEY( hstate_type, fc_type, output_type )    \
        (hstate_type | (fc_type << 8) | (output_type << 16))

    VSI_UNREFERENCED(param_size);

    output = (vsi_nn_kernel_tensor_t)param[3];

    for (i = 0; i < 2; i++)
    {
        input_attr[i] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[i] );
        CHECK_PTR_FAIL_GOTO( input_attr[i], "Create tensor attr buffer fail.", final );
    }

    output_attr[0] = vsi_nn_kernel_tensor_attr_create( output );
    CHECK_PTR_FAIL_GOTO( output_attr[0], "Create tensor attr buffer fail.", final );

    hstate_in_scale = input_attr[0]->scale;
    hstate_in_tail = 0 - (float)input_attr[0]->zero_point * hstate_in_scale;

    pack_key = _PACK_SELECT_KEY( input_attr[0]->dtype, input_attr[1]->dtype, output_attr[0]->dtype);

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((input_attr[0]->shape->data[0] + gpu_param.global_scale[0] - 1)
        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (input_attr[0]->shape->data[1] + gpu_param.global_scale[1] - 1)
        / gpu_param.global_scale[1];

    switch (pack_key)
    {
    case _PACK_SELECT_KEY(F16, F16, F16):
        {
            gpu_dp_inst_t uniExtractH4_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniF16PlusF16_0_4x4 = {{
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00110000, 0x00330022, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertF16_0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniExtractH4_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniF16PlusF16_0_4x4", &uniF16PlusF16_0_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertF16_0_4x4", &uniConvertF16_0_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case _PACK_SELECT_KEY(BF16, BF16, BF16):
        {
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

            status  = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8", &uniExtractOddData_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case _PACK_SELECT_KEY(U8,  F16, F16):
    case _PACK_SELECT_KEY(I8,  F16, F16):
    case _PACK_SELECT_KEY(I16, F16, F16):
        {
            gpu_dp_inst_t uniExtractH4_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniF16PlusF16_0_4x4 = {{
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00110000, 0x00330022, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertF16_0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniExtractH4_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniF16PlusF16_0_4x4", &uniF16PlusF16_0_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertF16_0_4x4", &uniConvertF16_0_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "hstate_in_scale", &hstate_in_scale);
            status |= vsi_nn_kernel_gpu_add_param(node, "hstate_in_tail", &hstate_in_tail);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        break;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    for (i = 0; i < 2; i++)
    {
        if (input_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &input_attr[i] );
        }
    }

    if (output_attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &output_attr[0] );
    }

    return status;
} /* _grucell_h_times_activation_r_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t  recurrent_activation
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e hstate_dtype;
    vsi_nn_kernel_dtype_e fc_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _grucell_h_times_activation_r_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _grucell_h_times_activation_r_kernel_map );
    vx_param_description_t * param_def  = _grucell_h_times_activation_r_kernel_param_def;
    vx_kernel_initialize_f  initializer = _grucell_h_times_activation_r_initializer;

    uint32_t key;
    uint32_t i;

    hstate_dtype  = vsi_nn_kernel_map_dtype( inputs[GRUCELL_ACT_H_STATE]->attr.dtype.vx_type );
    fc_dtype  = vsi_nn_kernel_map_dtype( inputs[GRUCELL_ACT_I_FC_Z]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[GRUCELL_ACT_OUT_OUTPUT]->attr.dtype.vx_type );

    key = GRUCELL_H_TIMES_ACTIVATION_R_HASH_KEY( hstate_dtype, fc_dtype, out_dtype, recurrent_activation );

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
        kernel->info.numParams   = _cnt_of_array( _grucell_h_times_activation_r_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );

    status = _query_kernel( kernel, inputs, outputs, recurrent_activation );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( grucell_h_times_activation_r, _setup )

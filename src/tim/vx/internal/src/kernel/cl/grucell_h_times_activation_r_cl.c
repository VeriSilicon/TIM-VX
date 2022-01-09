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
 */
typedef enum _grucell_nn_activation_type_e
{
    SIGMOID = VSI_NN_ACT_SIGMOID,
    HARD_SIGMOID = VSI_NN_ACT_HARD_SIGMOID,
} grucell_nn_activation_type_e;

#define _GRUCELL_H_TIMES_ACTIVATION_R_KERNEL_SOURCE      "grucell_h_times_activation_r"
#define _GRUCELL_H_TIMES_ACTIVATION_R_KERNEL_NAME        CVIVANTE_NAMESPACE("cl.grucell_h_times_activation_r")

// Add kernel hashtable here
#define GRUCELL_H_TIMES_ACTIVATION_R_HASH_KEY( HSTATE_DTYPE, IN_FC_DTYPE, OUT_TYPE, REC_ACT ) \
        (( HSTATE_DTYPE ) | ( IN_FC_DTYPE << 6 ) | ( OUT_TYPE << 12 ) | ( REC_ACT << 18 ))
#define PACK_KERNEL_MAP( HSTATE_DTYPE, IN_FC_DTYPE, OUT_TYPE, REC_ACT ) \
      { GRUCELL_H_TIMES_ACTIVATION_R_HASH_KEY( HSTATE_DTYPE, IN_FC_DTYPE, OUT_TYPE, REC_ACT ), \
        CVIVANTE_NAMESPACE("cl.grucell_h_times_activation_r_"#HSTATE_DTYPE"_"#IN_FC_DTYPE"to"#OUT_TYPE"_"#REC_ACT), \
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
    PACK_KERNEL_MAP( U8,  F32, F32, SIGMOID ),
    PACK_KERNEL_MAP( I32, F32, F32, SIGMOID ),
    PACK_KERNEL_MAP( F32, F32, F32, SIGMOID ),
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
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM  _cnt_of_array( _grucell_h_times_activation_r_kernel_param_def )
#define SCALAR_INPUT_SCALE      (4)
#define SCALAR_INPUT_TAIL       (5)
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
    vsi_nn_kernel_tensor_attr_t* output_attr;

    output = (vsi_nn_kernel_tensor_t)param[3];

    output_attr = vsi_nn_kernel_tensor_attr_create( output );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((output_attr->shape->data[0] + gpu_param.global_scale[0] - 1)
        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (output_attr->shape->data[1] + gpu_param.global_scale[1] - 1)
        / gpu_param.global_scale[1];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:

    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release( &output_attr );
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

    uint32_t key = 0;
    uint32_t i = 0;

    hstate_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    fc_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == hstate_dtype)
    {
        hstate_dtype = F32;
    }
    else if (I8 == hstate_dtype || I16 == hstate_dtype)
    {
        hstate_dtype = I32;
    }

    if (F16 == fc_dtype)
    {
        fc_dtype = F32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }
    else if (I8 == out_dtype || I16 == out_dtype)
    {
        out_dtype = I32;
    }

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
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
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
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float input_tail = -(float)vsi_nn_get_tensor_zero_point(inputs[0]) * input_scale;

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
            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input_scale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input_tail );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( grucell_h_times_activation_r, _setup )

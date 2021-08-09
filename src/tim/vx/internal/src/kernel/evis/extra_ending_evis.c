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


// Add kernel hashtable here
#define EXTRA_ENDING_HASH_KEY( OUT_DTYPE ) \
        ( ( OUT_DTYPE ) )
#define EXTRA_ENDING_KERNEL_MAP( OUT_DTYPE ) \
        { EXTRA_ENDING_HASH_KEY( OUT_DTYPE ), \
         CVIVANTE_NAMESPACE("evis.extra_ending_"#OUT_DTYPE), \
         "extra_ending" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _extra_ending_kernel_map[] =
{
    // Register kernel here
    EXTRA_ENDING_KERNEL_MAP( F16 ),
    EXTRA_ENDING_KERNEL_MAP( I16 ),
    EXTRA_ENDING_KERNEL_MAP( U8 ),
    EXTRA_ENDING_KERNEL_MAP( I8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _extra_ending_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _EXTRA_ENDING_PARAM_NUM  _cnt_of_array( _extra_ending_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_extra_ending_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr  = NULL;
    vsi_int_array_t * out_shape          = NULL;

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    out_shape = attr->shape;

    gpu_param.global_scale[0] = 8;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0];
    gpu_param.global_size[1] = out_shape->data[1];
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (attr)
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
        attr = NULL;
    }

    return status;
} /* _extra_ending_initializer() */


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
    vsi_nn_kernel_dtype_e out_dtype;
    uint32_t key = 0;
    uint32_t i = 0;

    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = EXTRA_ENDING_HASH_KEY( out_dtype );

    for ( i = 0; i < _cnt_of_array(_extra_ending_kernel_map); i ++ )
    {
        if ( _extra_ending_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_extra_ending_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _extra_ending_kernel_map[i].function_name );
        kernel->info.parameters  = _extra_ending_kernel_param_def;
        kernel->info.numParams   = _cnt_of_array( _extra_ending_kernel_param_def );
        kernel->info.initialize  = _extra_ending_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _extra_ending_kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _extra_ending_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_EXTRA_ENDING_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    uint32_t rank[3] = {0};
    int32_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    int32_t i = 0;

    vsi_nn_kernel_optimize_1d_tensor_shape( (const int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num,
        shapes[0], &rank[0]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const int32_t*)inputs[1]->attr.size, inputs[1]->attr.dim_num,
        shapes[1], &rank[1]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num,
        shapes[2], &rank[2]);

    for (i = 0; i < 2; i++)
    {
        reshape_tensors[i] = vsi_nn_reshape_tensor( graph,
            inputs[i], (uint32_t*)shapes[i], rank[i] );
    }
    reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
        outputs[0], (uint32_t*)shapes[2], rank[2] );

    if ( !vsi_nn_kernel_gpu_check_shape( (int32_t*)reshape_tensors[0]->attr.size,
        inputs[0]->attr.dim_num ) )
    {
        goto final;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            vx_border_t border;

            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U32 = 0;
            status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border));
            CHECK_STATUS_FAIL_GOTO( status, final );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _EXTRA_ENDING_PARAM_NUM,
                    reshape_tensors, input_num, &reshape_tensors[2], output_num );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, node_params, _EXTRA_ENDING_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, final );
        }
    }

final:
    for (i = 0; i < 3; i++)
    {
        vsi_safe_release_tensor(reshape_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( extra_ending, _setup )

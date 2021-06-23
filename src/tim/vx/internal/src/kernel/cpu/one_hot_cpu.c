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
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
 #define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.one_hot")


/*
 * Kernel params
 */
static vx_param_description_t _one_hot_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define INPUT_SCALAR_DEPTH          (2)
#define INPUT_SCALAR_ON_VALUE       (3)
#define INPUT_SCALAR_OFF_VALUE      (4)
#define INPUT_SCALAR_AXIS           (5)
#define _ONE_HOT_PARAM_NUM  _cnt_of_array( _one_hot_kernel_param_def )


/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_compute)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_IO_NUM] = { NULL };
    float * buffer[_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_IO_NUM] = { NULL };
    int32_t i = 0;
    int32_t j = 0;
    int32_t k = 0;
    int32_t index = 0;
    int32_t depth = 0;
    float on_value = 0;
    float off_value = 0;
    int32_t axis = 0;
    int32_t prefix_dim_size = 1;
    int32_t suffix_dim_size = 0;
    int32_t num_elements = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &depth);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[3], &on_value);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[4], &off_value);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    num_elements = (int32_t)vsi_nn_kernel_tensor_attr_get_size( attr[0] );
    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );
    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    axis = axis == -1 ? (int32_t)attr[0]->shape->size : (int32_t)attr[0]->shape->size - axis;

    for (i = 0; i < axis; i++)
    {
        prefix_dim_size *= attr[0]->shape->data[i];
    }

    suffix_dim_size = num_elements / prefix_dim_size;

    for (i = 0; i < prefix_dim_size; i++)
    {
        for (j = 0; j < depth; j++)
        {
            for (k = 0; k < suffix_dim_size; k++)
            {
                int32_t value = (int32_t)buffer[0][i * suffix_dim_size + k];
                buffer[1][index ++] = value == j ? on_value : off_value;
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );
final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if ( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR
    for ( i = 0; i < _IO_NUM; i ++ )
    {
        if ( buffer[i] )
        {
            free( buffer[i] );
            buffer[i] = NULL;
        }
    }

    return status;
} /* _compute() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _one_hot_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _one_hot_kernel_param_def );
    status = VSI_SUCCESS;

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
    vsi_nn_kernel_node_param_t node_params[_ONE_HOT_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t depth = vsi_nn_kernel_param_get_int32( params, "depth" );
    float on_value = vsi_nn_kernel_param_get_float32( params, "on_value" );
    float off_value = vsi_nn_kernel_param_get_float32( params, "off_value" );
    int32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ONE_HOT_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[INPUT_SCALAR_DEPTH] = vsi_nn_kernel_scalar_create(
                    graph, I32, &depth );
            node_params[INPUT_SCALAR_ON_VALUE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &on_value );
            node_params[INPUT_SCALAR_OFF_VALUE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &off_value );
            node_params[INPUT_SCALAR_AXIS] = vsi_nn_kernel_scalar_create(
                    graph, I32, &axis );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ONE_HOT_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, OnError );
        }
    }
OnError:
    if (node_params[INPUT_SCALAR_DEPTH])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_SCALAR_DEPTH] );
    }

    if (node_params[INPUT_SCALAR_ON_VALUE])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_SCALAR_ON_VALUE] );
    }

    if (node_params[INPUT_SCALAR_OFF_VALUE])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_SCALAR_OFF_VALUE] );
    }

    if (node_params[INPUT_SCALAR_AXIS])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_SCALAR_AXIS] );
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( one_hot, _setup )

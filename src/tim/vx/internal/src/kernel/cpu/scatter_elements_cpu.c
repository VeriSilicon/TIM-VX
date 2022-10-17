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
#define _ARG_NUM            (2)
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.gather_elements")


/*
 * Kernel params
 */
static vx_param_description_t _scatter_elements_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SCATTER_ELEMENTS_PARAM_NUM  _cnt_of_array( _scatter_elements_kernel_param_def )


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
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[3] = { NULL };
    int32_t* buffer_idx = NULL;
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    vsi_size_t a = 0;
    vsi_size_t o = 0;
    vsi_size_t i = 0;
    vsi_size_t outer_size[2] = {1, 1};
    vsi_size_t inner_size[2] = {1, 1};
    vsi_size_t axis_size[2] = {1, 1};
    int32_t axis = 0;
    int32_t reduction = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];
    tensors[3]  = (vsi_nn_kernel_tensor_t)param[3];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
    attr[3] = vsi_nn_kernel_tensor_attr_create( tensors[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );
    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[3] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &reduction);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer_idx = (int32_t*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], FALSE );
    CHECK_PTR_FAIL_GOTO( buffer_idx, "Create input1 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input0 buffer fail.", final );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memcpy( buffer[2], buffer[0], out_elements * sizeof(float) );

    axis_size[0] = attr[0]->shape->data[axis];
    axis_size[1] = attr[1]->shape->data[axis];
    for (i = 0; i < (vsi_size_t)axis; ++i)
    {
        inner_size[0] *= attr[0]->shape->data[i];
        inner_size[1] *= attr[1]->shape->data[i];
    }

    for (i = axis + 1; i < attr[1]->shape->size; ++i)
    {
        outer_size[0] *= attr[0]->shape->data[i];
        outer_size[1] *= attr[1]->shape->data[i];
    }

    for (o = 0; o < outer_size[1]; o++)
    {
        for (a = 0; a < axis_size[1]; a++)
        {
            for (i = 0; i < inner_size[1]; i++)
            {
                vsi_ssize_t index = 0;
                vsi_size_t index0 = (o * axis_size[1] + a) * inner_size[1] + i;
                vsi_size_t index1 = 1;

                index = (vsi_ssize_t)buffer_idx[index0];
                index1 = (o * axis_size[0] + index) * inner_size[0] + i;

                switch (reduction)
                {
                    case VSI_NN_REDUCTION_TYPE_NONE:
                        buffer[2][index1] = buffer[1][index0];
                        break;
                    case VSI_NN_REDUCTION_TYPE_ADD:
                        buffer[2][index1] += buffer[1][index0];
                        break;
                    case VSI_NN_REDUCTION_TYPE_MUL:
                        buffer[2][index1] *= buffer[1][index0];
                        break;
                    default:
                        break;
                }


            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
            buffer[2], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );
final:
    if ( buffer_idx )
    {
        free( buffer_idx );
    }
    for ( i = 0; i < 3; i ++ )
    {
        if ( buffer[i] )
        {
            free( buffer[i] );
        }
    }
    for ( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if (attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
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
    vsi_status status = VSI_SUCCESS;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _scatter_elements_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _scatter_elements_kernel_param_def );

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
    vsi_nn_kernel_node_param_t node_params[_SCATTER_ELEMENTS_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t reduction = vsi_nn_kernel_param_get_int32( params, "reduction" );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SCATTER_ELEMENTS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[4] = vsi_nn_kernel_scalar_create( graph, I32, &axis );
            node_params[5] = vsi_nn_kernel_scalar_create( graph, I32, &reduction );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SCATTER_ELEMENTS_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( scatter_elements, _setup )


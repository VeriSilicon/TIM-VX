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
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _CPU_IO_NUM         (_INPUT_NUM + _OUTPUT_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.logical_ops")

/*
 * Kernel params
 */
static vx_param_description_t _logical_ops_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _LOGICAL_OPS_PARAM_NUM  _cnt_of_array( _logical_ops_kernel_param_def )


static vsi_ssize_t _expand_offset
    (
    vsi_ssize_t index,
    vsi_size_t * shape, vsi_size_t rank,
    vsi_size_t * strides, vsi_size_t * out_shape
    )
{
    vsi_size_t i;
    vsi_ssize_t offset = 0;

    for( i = 0; i < rank && index; i ++ )
    {
        if( shape[i] == out_shape[i] )
        {
            offset += (vsi_ssize_t)strides[i] * ( index % out_shape[i] );
        }
        index /= out_shape[i];
    }
    return offset;
}

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
    vsi_nn_kernel_tensor_t input[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float *f32_in_buffer[_INPUT_NUM] = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM] = {NULL};
    vsi_size_t   in_stride_size[_INPUT_NUM][VSI_NN_MAX_DIM_NUM]   = {{1}};
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i;
    uint32_t  ops_type_int = 0;
    vsi_nn_logical_ops_type_t  ops_type = VSI_NN_LOGICAL_OR;
    /* prepare data */
    for(i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        vsi_nn_kernel_tensor_attr_get_stride( in_attr[i], in_stride_size[i] );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );
    }
    for(i = 0; i < _OUTPUT_NUM; i ++)
    {
        output[i] = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );
        vsi_nn_kernel_tensor_attr_get_stride( out_attr[i], out_stride_size[i] );
        out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
        out_bytes[i] = out_elements[i] * sizeof(float);
        f32_out_buffer[i] = (float *)malloc( out_bytes[i] );
        CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
        memset( f32_out_buffer[i], 0, out_bytes[i] );
    }
    vsi_nn_kernel_scalar_read_uint32((vsi_nn_kernel_scalar_t)param[_CPU_IO_NUM], &(ops_type_int));
    ops_type = (vsi_nn_logical_ops_type_t)ops_type_int;
    if (!(VSI_NN_LOGICAL_OR == ops_type || VSI_NN_LOGICAL_AND == ops_type || VSI_NN_LOGICAL_XOR == ops_type))
    {
        status = VSI_FAILURE;
        goto final;
    }

    for (i = 0; i < out_elements[0]; i++)
    {
        vsi_ssize_t  in0_offset = 0;
        vsi_ssize_t  in1_offset = 0;
        vsi_ssize_t  in0 = 0;
        vsi_ssize_t  in1 = 0;


        in0_offset = _expand_offset( i, in_attr[0]->shape->data, (vsi_size_t)in_attr[0]->shape->size,
                in_stride_size[0], out_attr[0]->shape->data );
        in1_offset = _expand_offset( i, in_attr[1]->shape->data, (vsi_size_t)in_attr[1]->shape->size,
                in_stride_size[1], out_attr[0]->shape->data );
        in0 = (!!(f32_in_buffer[0][in0_offset]));
        in1 = (!!(f32_in_buffer[1][in1_offset]));
        if (VSI_NN_LOGICAL_OR == ops_type)
        {
            f32_out_buffer[0][i] = (float)(in0 || in1);
        }
        else if (VSI_NN_LOGICAL_AND == ops_type)
        {
            f32_out_buffer[0][i] = (float)(in0 && in1);
        }
        else if (VSI_NN_LOGICAL_XOR == ops_type)
        {
            f32_out_buffer[0][i] = (float)(in0 ^ in1);
        }
    }

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                f32_out_buffer[i], out_elements[i] );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    for (i = 0; i < _INPUT_NUM; i++)
    {
        if (f32_in_buffer[i])
        {
            free(f32_in_buffer[i]);
            f32_in_buffer[i] = NULL;
        }

        if (in_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &in_attr[i] );
        }
    }
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        if (f32_out_buffer[i])
        {
            free(f32_out_buffer[i]);
            f32_out_buffer[i] = NULL;
        }

        if (out_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &out_attr[i] );
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
    kernel->info.parameters  = _logical_ops_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _logical_ops_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_LOGICAL_OPS_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    uint32_t ops_type  = vsi_nn_kernel_param_get_int32( params, "ops_type" );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _LOGICAL_OPS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[_CPU_IO_NUM] = vsi_nn_kernel_scalar_create( graph, U32, &ops_type );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _LOGICAL_OPS_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( logical_ops, _setup )

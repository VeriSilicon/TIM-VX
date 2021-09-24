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
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.conv1d_ovxlib")

/*
 * Kernel params
 */
static vx_param_description_t _conv1d_ovxlib_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _CONV1D_OVXLIB_PARAM_NUM  _cnt_of_array( _conv1d_ovxlib_kernel_param_def )
#define _IO_COUNT       (4)

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
    int i = 0;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_IO_COUNT] = { NULL };
    vsi_nn_kernel_tensor_attr_t* attr[_IO_COUNT] = { NULL };
    float* buffer[_IO_COUNT] = { NULL };
    int32_t stride = 0;
    int32_t pad_front = 0;
    int32_t pad_end = 0;
    int32_t dilation = 0;
    int32_t overflow_policy = 0;
    int32_t rounding_policy = 0;
    int32_t down_scale_size_rounding = 0;

    tensors[0] = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1] = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2] = (vsi_nn_kernel_tensor_t)param[2];
    tensors[3] = (vsi_nn_kernel_tensor_t)param[3];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    attr[3] = vsi_nn_kernel_tensor_attr_create( tensors[3] );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );
    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input buffer fail.", final );
    buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    buffer[3] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[3], attr[3], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create input buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &stride);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &pad_front);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &pad_end);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &dilation);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &overflow_policy);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &rounding_policy);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &down_scale_size_rounding);
    CHECK_STATUS_FAIL_GOTO(status, final);

    {
        vsi_ssize_t batch = attr[0]->shape->data[2];
        vsi_ssize_t input_channel = attr[0]->shape->data[1];
        vsi_ssize_t input_height = attr[0]->shape->data[0];
        vsi_ssize_t kernel_size = attr[1]->shape->data[0];
        vsi_ssize_t output_channel = attr[1]->shape->data[2];
        vsi_ssize_t output_height = attr[3]->shape->data[0];
        vsi_ssize_t batch_index = 0;
        vsi_ssize_t input_channel_index = 0;
        vsi_ssize_t output_channel_index = 0;
        vsi_ssize_t output_h_index = 0;

        for(batch_index = 0; batch_index < batch; batch_index++)
        {
            float* per_batch_input = buffer[0] + batch_index * input_channel * input_height;
            float* per_batch_output = buffer[3] + batch_index * output_channel * output_height;
            for(output_channel_index = 0; output_channel_index < output_channel; output_channel_index++)
            {
                float* filter = buffer[1] + output_channel_index * input_channel * kernel_size;
                for(output_h_index = 0; output_h_index < output_height; output_h_index++)
                {
                    float output_value = 0.;
                    float* current_value_ptr = per_batch_input + output_h_index * stride;

                    for(input_channel_index = 0; input_channel_index < input_channel; input_channel_index++)
                    {
                        int k = 0;
                        int32_t index = 0;
                        for(k = 0; k < kernel_size; k++)
                        {
                            float w = *(filter + input_channel_index * kernel_size + k);
                            float v = *(current_value_ptr + input_channel_index * input_height + index);

                            output_value += w * v;
                            index += dilation;
                        }
                    }

                    if(buffer[2])
                    {
                        output_value += buffer[2][output_channel_index];
                    }

                    *(per_batch_output + output_channel_index * output_height + output_h_index) = output_value;
                }
            }
        }
        status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
                buffer[3], batch * output_channel * output_height );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    for( i = 0; i < _IO_COUNT; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
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
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _conv1d_ovxlib_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _conv1d_ovxlib_kernel_param_def );

    return VSI_SUCCESS;
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
    vsi_nn_kernel_node_param_t node_params[_CONV1D_OVXLIB_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int j = 0;

    int32_t stride = vsi_nn_kernel_param_get_int32( params, "stride" );
    int32_t pad_front = vsi_nn_kernel_param_get_int32( params, "pad_front" );
    int32_t pad_end = vsi_nn_kernel_param_get_int32( params, "pad_end" );
    int32_t dilation = vsi_nn_kernel_param_get_int32( params, "dilation" );
    int32_t overflow_policy = vsi_nn_kernel_param_get_int32( params, "overflow_policy" );
    int32_t rounding_policy = vsi_nn_kernel_param_get_int32( params, "rounding_policy" );
    int32_t down_scale_size_rounding = vsi_nn_kernel_param_get_int32( params, "down_scale_size_rounding" );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CONV1D_OVXLIB_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            j = (int)(input_num + output_num);
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &stride );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &pad_front );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &pad_end );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &dilation );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &overflow_policy );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &rounding_policy );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &down_scale_size_rounding );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CONV1D_OVXLIB_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( conv1d_ovxlib, _setup )

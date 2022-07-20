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
#define _CPU_ARG_NUM            (8)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (2)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.maxpoolwithargmax")

#define FP32_MIN                -3.4e38

/*
 * Kernel params
 */
static vx_param_description_t _maxpoolwithargmax_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
    // Add kererl parameters here
};
#define _MAXPOOLWITHARGMAX_PARAM_NUM  _cnt_of_array( _maxpoolwithargmax_kernel_param_def )

/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_maxpoolwithargmax_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    int32_t ksize_x = 0, ksize_y = 0, stride_x = 0, stride_y = 0;
    int32_t pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
    int32_t i = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &ksize_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &ksize_y);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &stride_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &stride_y);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &pad_left);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &pad_right);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &pad_top);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &pad_bottom);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset( buffer[2], 0, out_elements * sizeof(float) );

    {
        int32_t dims_num = (int32_t)attr[1]->shape->size;
        int32_t batch    = dims_num > 3 ? (int32_t)attr[1]->shape->data[3] : 1;
        int32_t depth    = dims_num > 2 ? (int32_t)attr[1]->shape->data[2] : 1;
        int32_t height_o = (int32_t)attr[1]->shape->data[1];
        int32_t width_o  = (int32_t)attr[1]->shape->data[0];
        int32_t width    = (int32_t)attr[0]->shape->data[0];
        int32_t height   = (int32_t)attr[0]->shape->data[1];
        int32_t b = 0, d = 0, j = 0;
        int32_t output_base = 0;
        int32_t input_base  = 0;

        for (b = 0; b < batch; b++)
        {
            for (d = 0; d < depth; d++)
            {
                output_base = b * depth * height_o * width_o + d * height_o * width_o;
                input_base = b * depth * height * width + d * height * width;
                for (j = 0; j < height_o; j++)
                {
                    for (i = 0; i < width_o; i++)
                    {
                        int32_t hstart = j * stride_y - pad_top;
                        int32_t wstart = i * stride_x - pad_left;
                        int32_t hend = vsi_nn_min(hstart + ksize_y, height);
                        int32_t wend = vsi_nn_min(wstart + ksize_x, width);
                        int32_t pool_index = output_base + j * width_o + i;
                        int32_t h = 0, w = 0;
                        int32_t index_max = 0;
                        float   value_max = (float)FP32_MIN;

                        hstart = vsi_nn_max(hstart, 0);
                        wstart = vsi_nn_max(wstart, 0);

                        for (h = hstart; h < hend; ++ h)
                        {
                            for (w = wstart; w < wend; ++ w)
                            {
                                int32_t index = input_base + h * width + w;
                                float data = buffer[0][index];

                                if (data > value_max)
                                {
                                    value_max = data;
                                    index_max = index;
                                }
                            }
                        }
                        buffer[1][pool_index] = value_max;
                        buffer[2][pool_index] = (float)index_max;
                    }
                }
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    status |= vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
            buffer[2], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for ( i = 0; i < _CPU_IO_NUM; i ++ )
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
} /* _maxpoolwithargmax_exec() */

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
    kernel->info.function    = _maxpoolwithargmax_exec;
    kernel->info.parameters  = _maxpoolwithargmax_kernel_param_def;
    kernel->info.numParams   = _MAXPOOLWITHARGMAX_PARAM_NUM;
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
    vsi_nn_kernel_node_param_t node_params[_MAXPOOLWITHARGMAX_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    int32_t ksize_x    = vsi_nn_kernel_param_get_int32(params, "ksize_x");
    int32_t ksize_y    = vsi_nn_kernel_param_get_int32(params, "ksize_y");
    int32_t stride_x   = vsi_nn_kernel_param_get_int32(params, "stride_x");
    int32_t stride_y   = vsi_nn_kernel_param_get_int32(params, "stride_y");
    int32_t pad_left   = vsi_nn_kernel_param_get_int32(params, "pad_left");
    int32_t pad_right  = vsi_nn_kernel_param_get_int32(params, "pad_right");
    int32_t pad_top    = vsi_nn_kernel_param_get_int32(params, "pad_top");
    int32_t pad_bottom = vsi_nn_kernel_param_get_int32(params, "pad_bottom");

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            int32_t index = 3;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _MAXPOOLWITHARGMAX_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_right );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_bottom );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _MAXPOOLWITHARGMAX_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( maxpoolwithargmax, _setup )


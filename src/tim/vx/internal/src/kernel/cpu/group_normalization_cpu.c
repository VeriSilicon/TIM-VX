/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _CPU_ARG_NUM            (2)
#define _CPU_INPUT_NUM          (3)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.group_norm")

DEF_KERNEL_EXECUTOR(_group_norm_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t spaceOrg = 0;
    float eps = .0f;

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

    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[4], &eps);
    CHECK_STATUS_FAIL_GOTO(status, final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &spaceOrg);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input1 buffer fail.", final );

    buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create input1 buffer fail.", final );

    buffer[3] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create output buffer fail.", final );
    memset( buffer[3], 0, out_elements * sizeof(float) );

    {
        vsi_size_t b = 0, c = 0;
        vsi_size_t height = attr[0]->shape->data[1];
        vsi_size_t width = attr[0]->shape->data[0];
        vsi_size_t ch = attr[0]->shape->size > 2 ? attr[0]->shape->data[2] : 1;
        vsi_size_t bh = attr[0]->shape->size > 3 ? attr[0]->shape->data[3] : 1;
        vsi_size_t spatial = height * width;

        for (b = 0; b < bh; b++)
        {
            for (c = 0; c < ch; c++)
            {
                vsi_size_t page = c * spatial + b * (spatial * ch);
                vsi_size_t paraIdx = c * attr[1]->shape->data[0];
                float sum = .0f;
                float sumsq = .0f;
                float mean = .0f;
                float vari = .0f;
                float data = 0;

                for (i = 0; i < spatial; i++)
                {
                    vsi_size_t index = page + i;
                    sum += buffer[0][index];
                }

                mean = sum / spatial;
                for (i = 0; i < spatial; i++)
                {
                    vsi_size_t index = page + i;
                    data = buffer[0][index] - mean;
                    sumsq += data * data;
                }

                vari = sumsq / spatial;
                vari = (float)(1.0 / sqrtf(vari + eps));

                for (i = 0; i < spatial; i++)
                {
                    float normVal = 0;
                    vsi_size_t index = page + i;
                    vsi_size_t tmpIdx = paraIdx + i / spaceOrg;
                    float scaleVal = buffer[2][tmpIdx];
                    float biasVal = buffer[1][tmpIdx];

                    data = buffer[0][index] - mean;
                    normVal = data * vari * scaleVal + biasVal;
                    buffer[3][index] = normVal;
                }
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
            buffer[3], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if ( buffer[i] )
        {
            free( buffer[i] );
        }
    }
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if (attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }
    return status;
} /* _group_norm_exec() */
/*
 * Kernel params
 */
static vx_param_description_t _group_normalization_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GROUP_NORMALIZATION_PARAM_NUM  _cnt_of_array( _group_normalization_kernel_param_def )

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _group_norm_exec;
    kernel->info.parameters  = _group_normalization_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _group_normalization_kernel_param_def );

    return VSI_SUCCESS;
} /* _query_kernel() */

static int32_t _optimize_gn_shape_cpu
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t group_size,
    int32_t group_num,
    vsi_size_t* opt_shape
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_size_t group_shape[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_rank = 0;
    group_shape[0] = inputs[0]->attr.size[0];
    group_shape[1] = inputs[0]->attr.size[1];
    group_shape[2] = group_size;

    vsi_nn_kernel_optimize_element_shape(group_shape, 3, opt_shape, &new_rank );

    if (new_rank == 2)
    {
        opt_shape[2] = group_num;
        opt_shape[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    }
    else
    {
        status = VSI_FAILURE;
    }

    return status;
}

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
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_size_t new_shape[VSI_NN_MAX_DIM_NUM] = { 1, 1, 1, 1 };
    int32_t group_num  = vsi_nn_kernel_param_get_int32( params, "group_num" );
    vsi_size_t group_size  = inputs[0]->attr.size[2] / group_num;
    int32_t spaceOrg = (int32_t)(inputs[0]->attr.size[0] * inputs[0]->attr.size[1]);

    status = _optimize_gn_shape_cpu(inputs, group_size, group_num, new_shape);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape, 4);
    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape, 4);

    status = _query_kernel( inputs, outputs, kernel );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            float eps  = vsi_nn_kernel_param_get_float32( params, "eps" );
            uint32_t index = 0;
            /* Set inputs and outputs */
            backend_params[index++] = rs_input;
            backend_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
            backend_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
            backend_params[index++] = rs_output;
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &spaceOrg );

            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
final:
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( group_norm, _setup )

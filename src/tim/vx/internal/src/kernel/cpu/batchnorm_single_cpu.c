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
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

#define _CPU_ARG_NUM            (1)
#define _CPU_INPUT_NUM          (5)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("batch_norm_sw")

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

DEF_KERNEL_EXECUTOR(_batch_norm_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    vsi_size_t out_elements = 0;
    vsi_size_t stride_size[_CPU_INPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    float eps = 0.f;

    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[param_size - 1], &eps);
    CHECK_STATUS_FAIL_GOTO(status, final );

    for ( i = 0;  i < _CPU_INPUT_NUM;  i++)
    {
        tensors[i]  = (vsi_nn_kernel_tensor_t)param[i];
        attr[i] = vsi_nn_kernel_tensor_attr_create( tensors[i] );

        vsi_nn_kernel_tensor_attr_get_stride( attr[i], stride_size[i] );
        buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[i], attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( buffer[i], "Create input buffer fail.", final );
    }

    tensors[5]  = (vsi_nn_kernel_tensor_t)param[5];
    attr[5] = vsi_nn_kernel_tensor_attr_create( tensors[5] );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[5] );

    buffer[5] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[5], "Create output buffer fail.", final );
    memset( buffer[5], 0, out_elements * sizeof(float) );

    for( i = 0; i < out_elements; i ++ )
    {
        vsi_ssize_t in_offset[5] = {0};
        int32_t j = 0;
        float src = 0.f;
        float mean = 0.f;
        float variance = 0.f;
        float beta = 0.f;
        float gamma = 0.f;

        for ( j = 0; j < 5; j++)
        {
            in_offset[j] = _expand_offset( i, attr[j]->shape->data, (vsi_size_t)attr[j]->shape->size,
                    stride_size[j], attr[5]->shape->data );
        }

        src = buffer[0][in_offset[0]];
        mean = buffer[1][in_offset[1]];
        variance = buffer[2][in_offset[2]];
        gamma = buffer[3][in_offset[3]];
        beta = buffer[4][in_offset[4]];


        buffer[5][i] = (src - mean) * gamma/ sqrtf(variance + eps) + beta;
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[5], attr[5],
            buffer[5], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
    }
    return status;
} /* _batch_norm_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_INPUT_EPS          (6)

static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _batch_norm_exec,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
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
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float eps = 0;

    eps = vsi_nn_kernel_param_get_float32(params, "eps");

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            /* Pass parameters to node. */
            backend_params[SCALAR_INPUT_EPS] = vsi_nn_kernel_scalar_create(
                    graph, F32, &eps );

            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &backend_params[SCALAR_INPUT_EPS] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( batchnorm_single, _setup )


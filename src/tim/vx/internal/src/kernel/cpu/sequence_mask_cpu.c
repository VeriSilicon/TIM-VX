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
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "libnnext/vsi_nn_vxkernel.h"

__BEGIN_DECLS

#define _CPU_ARG_NUM            (1)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("sequence_mask_sw")

DEF_KERNEL_EXECUTOR(_sequence_mask_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer_in = NULL;
    float * buffer = NULL;
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    buffer_in = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer_in, "Create input0 buffer fail.", final );

    buffer = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer, "Create output buffer fail.", final );
    memset( buffer, 0, out_elements * sizeof(float) );

    {
        vsi_size_t j = 0;
        vsi_size_t height = attr[1]->shape->data[1];
        vsi_size_t width = attr[1]->shape->data[0];

        for(j = 0; j < height; j++)
        {
            vsi_size_t idx_in = (vsi_size_t)buffer_in[j];
            vsi_size_t out_offset = j * width;
            idx_in = idx_in > width ? width : idx_in;
            for(i = 0; i < idx_in; i++)
            {
                buffer[out_offset + i] = 1;
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer, out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (buffer_in)
    {
        free( buffer_in );
    }
    if (buffer)
    {
        free( buffer );
    }
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
    }
    return status;
} /* _sequence_mask_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};


static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _sequence_mask_exec,
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

static int32_t _optimize_mask_shape
    (
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    int32_t max_len,
    vsi_size_t* opt_shape_in,
    vsi_size_t* opt_shape_out
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_size_t out_size = 1;
    uint32_t i = 0;
    opt_shape_in[0] = 1;
    opt_shape_in[1] = 1;
    for(i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        opt_shape_in[0] *= inputs[0]->attr.size[i];
    }

    for(i = 0; i < outputs[0]->attr.dim_num; i++)
    {
        out_size *= outputs[0]->attr.size[i];
    }

    opt_shape_out[0] = max_len;
    opt_shape_out[1] = out_size / max_len;

    if (out_size % max_len != 0)
    {
        return VSI_FAILURE;
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
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_size_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }};
    int32_t max_len  = vsi_nn_kernel_param_get_int32( params, "max_len" );

    status = _optimize_mask_shape(inputs, outputs, max_len, new_shape[0], new_shape[1]);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape[0], 2);
    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape[1], 2);

    status = _query_kernel( inputs, outputs, kernel );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            uint32_t index = 0;
            /* Pass parameters to node. */
            backend_params[index++] = rs_input;
            backend_params[index++] = rs_output;
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &max_len );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &backend_params[2] );
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

REGISTER_BACKEND_CPU( sequence_mask, _setup )


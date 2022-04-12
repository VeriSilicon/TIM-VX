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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _CPU_ARG_NUM            (4)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.gather")

DEF_KERNEL_EXECUTOR(_gather_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[2] = { NULL };
    uint32_t* buffer_idx = NULL;
    size_t in_elements = 0, out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    vsi_size_t i = 0, j = 0, b = 0;
    int32_t block_size = 1, block_num = 1, axis_num = 0, batch_dims = 0;
    vsi_size_t indices_num = 1, batch = 1, in_stride = 1, out_stride = 1;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    in_elements = vsi_nn_kernel_tensor_attr_get_size( attr[0] );
    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[2] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &block_size);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &block_num);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &axis_num);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &batch_dims);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer_idx = (uint32_t*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], FALSE );
    CHECK_PTR_FAIL_GOTO( buffer_idx, "Create input1 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    {
        for (i = 0; i < attr[1]->shape->size - (vsi_size_t)batch_dims; i++)
        {
            indices_num *= attr[1]->shape->data[i];
        }

        for (; i < attr[1]->shape->size; i++)
        {
            batch *= attr[1]->shape->data[i];
        }

        for (i = 0; i < attr[0]->shape->size - (vsi_size_t)batch_dims; i++)
        {
            in_stride *= attr[0]->shape->data[i];
        }

        for (i = 0; i < attr[2]->shape->size - (vsi_size_t)batch_dims; i++)
        {
            out_stride *= attr[2]->shape->data[i];
        }

        for (b = 0; b < batch; b++)
        {
            for (i = 0; i < (vsi_size_t)block_num; i++)
            {
                for (j = 0; j < indices_num; j++)
                {
                    uint32_t indice = buffer_idx[j + indices_num * b];
                    vsi_size_t in_index = (i * axis_num + indice) * block_size + b * in_stride;
                    if (in_index < in_elements)
                    {
                        vsi_size_t out_index = (i * indices_num + j) * block_size + b * out_stride;
                        memcpy(&(buffer[1][out_index]), &(buffer[0][in_index]), block_size * sizeof(float));
                    }
                    else
                    {
                        status = VX_FAILURE;
                        CHECK_STATUS_FAIL_GOTO( status, final );
                    }
                }
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
            buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if ( buffer_idx )
    {
        free( buffer_idx );
    }
    for ( i = 0; i < 2; i ++ )
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
} /* _gather_exec() */
/*
 * Kernel params
 */
static vx_param_description_t _gather_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GATHER_PARAM_NUM  _cnt_of_array( _gather_kernel_param_def )

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _gather_exec;
    kernel->info.parameters  = _gather_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _gather_kernel_param_def );

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
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( inputs, outputs, kernel );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 3;
            int32_t block_size   = vsi_nn_kernel_param_get_int32( params, "block_size" );
            int32_t block_num    = vsi_nn_kernel_param_get_int32( params, "block_num" );
            int32_t axis_num     = vsi_nn_kernel_param_get_int32( params, "axis_num" );
            int32_t batch_dims   = vsi_nn_kernel_param_get_int32( params, "batch_dims" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );

            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_num );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_num );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &batch_dims );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[3] );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
            vsi_nn_kernel_scalar_release( &backend_params[6] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( gather, _setup )

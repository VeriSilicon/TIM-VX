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
#define _CPU_ARG_NUM            (3)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.cumsum")

DEF_KERNEL_EXECUTOR(_cumsum_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[2] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    int32_t i = 0;
    int32_t axisSize = 1, innerSize = 1, outerSize = 1;
    int32_t axis = 0, exclusive = 0, reverse = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &exclusive);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &reverse);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    {
        int32_t  dims_num  = (int32_t)attr[1]->shape->size;
        int32_t  inner     = 0;
        int32_t  outer     = 0;

        for(i = 0; i < axis; ++i)
        {
            innerSize *= (int32_t)attr[0]->shape->data[i];
        }

        axisSize = (int32_t)attr[0]->shape->data[i++];

        for(; i < dims_num; ++i)
        {
            outerSize *= (int32_t)attr[0]->shape->data[i];
        }

        for ( outer = 0; outer < outerSize; ++outer)
        {
            for ( inner = 0; inner < innerSize; ++inner)
            {
                float sum = .0f;

                if (exclusive && reverse)
                {
                    int32_t idx_out = (outer * axisSize + axisSize - 1) * innerSize + inner;
                    buffer[1][idx_out] = sum;
                    for (i = axisSize - 1; i > 0; i--)
                    {
                        int32_t idx = (outer * axisSize + i) * innerSize + inner;
                        float value = buffer[0][idx];
                        idx_out = (outer * axisSize + i - 1) * innerSize + inner;
                        sum += value;
                        buffer[1][idx_out] = sum;
                    }
                }
                else if (exclusive)
                {
                    int32_t idx_out = outer * axisSize * innerSize + inner;
                    buffer[1][idx_out] = sum;
                    for (i = 0; i < axisSize - 1; ++i)
                    {
                        int32_t idx = (outer * axisSize + i) * innerSize + inner;
                        float value = buffer[0][idx];
                        idx_out = (outer * axisSize + i + 1) * innerSize + inner;
                        sum += value;
                        buffer[1][idx_out] = sum;
                    }
                }
                else if (reverse)
                {
                    for (i = axisSize - 1; i >= 0; i--)
                    {
                        int32_t idx = (outer * axisSize + i) * innerSize + inner;
                        float value = buffer[0][idx];
                        sum += value;
                        buffer[1][idx] = sum;
                    }
                }
                else
                {
                    for (i = 0; i < axisSize; ++i)
                    {
                        // i * innerSize + inner + outer * innerSize * axisSize
                        int32_t idx = (outer * axisSize + i) * innerSize + inner;
                        float value = buffer[0][idx];
                        sum += value;
                        buffer[1][idx] = sum;
                    }
                }
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
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
} /* _cumsum_exec() */
/*
 * Kernel params
 */
static vx_param_description_t _cumsum_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CUMSUM_PARAM_NUM  _cnt_of_array( _cumsum_kernel_param_def )

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _cumsum_exec;
    kernel->info.parameters  = _cumsum_kernel_param_def;
    kernel->info.numParams   = _CUMSUM_PARAM_NUM;

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
            uint32_t index = 2;
            int32_t axis      = vsi_nn_kernel_param_get_int32( params, "axis" );
            int32_t exclusive = vsi_nn_kernel_param_get_int32( params, "exclusive" );
            int32_t reverse   = vsi_nn_kernel_param_get_int32( params, "reverse" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );

            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &exclusive );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[2] );
            vsi_nn_kernel_scalar_release( &backend_params[3] );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( cumsum, _setup )

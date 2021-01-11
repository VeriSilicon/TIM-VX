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
#include "kernel/vsi_nn_kernel.h"
#include "client/vsi_nn_vxkernel.h"

__BEGIN_DECLS

#define _CPU_ARG_NUM            (0)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("tile_sw")

void copyMultipleTimes(const float* in_data, int32_t in_size, int32_t multiplier, float* out_data)
{
    int i = 0;

    for ( i = 0; i < multiplier; ++i)
    {
        memcpy(out_data, in_data, in_size * sizeof(float));
        out_data += in_size;
    }
}

void tileOneDimension(const vsi_int_array_t* input_shape, const float* in_data,
                      const uint32_t* multipliers, float* out_data, int dimension,
                      int *stride_size, int *tiled_stride_size)
{
    int i = 0;
    const int dimension_size = input_shape->data[dimension];
    int total_stride_size = 0, total_tiled_stride_size = 0;
    const float* copy_from_data = in_data;
    float* copy_to_data = out_data;

    if (dimension == 0)
    {
        copyMultipleTimes(in_data, dimension_size, multipliers[dimension], out_data);
        *stride_size = dimension_size;
        *tiled_stride_size = dimension_size * multipliers[dimension];
        return ;
    }

    for (i = 0; i < dimension_size; ++i)
    {
        tileOneDimension(
                input_shape, copy_from_data, multipliers, copy_to_data, dimension - 1, stride_size, tiled_stride_size);
        copy_from_data += *stride_size;
        copy_to_data += *tiled_stride_size;
        total_stride_size += *stride_size;
        total_tiled_stride_size += *tiled_stride_size;
    }

    copyMultipleTimes(out_data, total_tiled_stride_size, multipliers[dimension] - 1,
                      out_data + total_tiled_stride_size);

    *stride_size = total_stride_size;
    *tiled_stride_size = total_tiled_stride_size * multipliers[dimension];
    return ;
}


DEF_KERNEL_EXECUTOR(_tile_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    uint32_t multiples[VSI_NN_MAX_DIM_NUM] = {0};
    int stride_size = 0, tiled_stride_size = 0;


    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    for (i = 0; i < attr[0]->shape->size; i++)
    {
        multiples[i] = attr[1]->shape->data[i] / attr[0]->shape->data[i];
    }

    tileOneDimension(attr[0]->shape, buffer[0], multiples, buffer[1],
        (int32_t)attr[0]->shape->size - 1, &stride_size, &tiled_stride_size);

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
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
} /* _tile_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};


static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _tile_exec,
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
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

REGISTER_BACKEND_CPU( tile, _setup )

__END_DECLS




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
#include "cpu_backend/npuref_interface.h"

__BEGIN_DECLS

typedef enum
{
    PARAM_INPUT = 0,
    PARAM_KERNEL,
    PARAM_BIAS,
    PARAM_OUTPUT,
    PARAM_STRIDE_0,
    PARAM_STRIDE_1,
    PARAM_PAD_0,
    PARAM_PAD_1,
    PARAM_PAD_2,
    PARAM_PAD_3,
    PARAM_DILATION_0,
    PARAM_DILATION_1,
    PARAM_MULTIPLIER,
    PARAM_NUM
} param_index_e;
/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.cpu_backend_conv2d")
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
/*
 * Kernel params
 */
static vx_param_description_t _cpu_backend_conv2d_kernel_param_def[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
};
#define _CPU_BACKEND_CONV2D_PARAM_NUM  _cnt_of_array( _cpu_backend_conv2d_kernel_param_def )


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
    vsi_nn_kernel_tensor_attr_t * attr[_IO_NUM] = { NULL };
    int32_t strides[2];
    int32_t pad[4];
    int32_t dilation[2];
    void * buffer[_IO_NUM] = { NULL };
    int32_t i = 0;
    vsi_nn_kernel_tensor_t tensors[_IO_NUM] = { NULL };
    size_t out_elements = 0;

    tensors[0] = (vsi_nn_kernel_tensor_t)param[PARAM_INPUT];
    tensors[1] = (vsi_nn_kernel_tensor_t)param[PARAM_KERNEL];
    tensors[2] = (vsi_nn_kernel_tensor_t)param[PARAM_BIAS];
    tensors[3] = (vsi_nn_kernel_tensor_t)param[PARAM_OUTPUT];
    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    if ( param[PARAM_BIAS] )
    {
        attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
        CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
    }
    attr[3] = vsi_nn_kernel_tensor_attr_create( tensors[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );
    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[3] );

    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_STRIDE_0], &strides[0] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_STRIDE_0], &strides[1] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_PAD_0], &pad[0] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_PAD_1], &pad[1] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_PAD_2], &pad[2] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_PAD_2], &pad[3] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_DILATION_0], &dilation[0] );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[PARAM_DILATION_1], &dilation[1] );
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], FALSE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], FALSE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create kernel buffer fail.", final );
    if ( param[PARAM_BIAS] )
    {
        buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], FALSE );
        CHECK_PTR_FAIL_GOTO( buffer[2], "Create bias buffer fail.", final );
    }
    buffer[3] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[3], attr[3], FALSE );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create output buffer fail.", final );

    npuref_interface_quant_conv2d(buffer[0], attr[0],
        buffer[1], attr[1], buffer[2],
        pad, strides, dilation, attr[3], buffer[3]);

    status = vsi_nn_kernel_tensor_write( tensors[3], attr[3],
        buffer[3], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for ( i = 0; i < _IO_NUM; i ++ )
    {
        if ( attr[i] )
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
        if ( buffer[i] )
        {
            free( buffer[i] );
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
    kernel->info.parameters  = _cpu_backend_conv2d_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _cpu_backend_conv2d_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_CPU_BACKEND_CONV2D_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    size_t size = 0;
    int32_t* stride = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "stride", &size);
    int32_t* pad = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "pad", &size);
    int32_t* dilation = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "dilation", &size);
    int32_t multiplier = vsi_nn_kernel_param_get_int32(params, "multiplier");

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CPU_BACKEND_CONV2D_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[4] = vsi_nn_kernel_scalar_create( graph, I32, &stride[0] );
            node_params[5] = vsi_nn_kernel_scalar_create( graph, I32, &stride[1] );
            node_params[6] = vsi_nn_kernel_scalar_create( graph, I32, &pad[0] );
            node_params[7] = vsi_nn_kernel_scalar_create( graph, I32, &pad[1] );
            node_params[8] = vsi_nn_kernel_scalar_create( graph, I32, &pad[2] );
            node_params[9] = vsi_nn_kernel_scalar_create( graph, I32, &pad[3] );
            node_params[10] = vsi_nn_kernel_scalar_create( graph, I32, &dilation[0] );
            node_params[11] = vsi_nn_kernel_scalar_create( graph, I32, &dilation[1] );
            node_params[12] = vsi_nn_kernel_scalar_create( graph, I32, &multiplier );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CPU_BACKEND_CONV2D_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( cpu_backend_conv2d, _setup )


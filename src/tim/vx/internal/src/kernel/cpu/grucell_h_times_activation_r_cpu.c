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
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.grucell_h_times_activation_r")


/*
 * Kernel params
 */
static vx_param_description_t _grucell_h_times_activation_r_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },  /*recurrent_activation*/
    // Add kererl parameters here
};
#define _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM  _cnt_of_array( _grucell_h_times_activation_r_kernel_param_def )
#define SCALAR_R_ACTIVATION        (4)

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
    vsi_nn_kernel_tensor_t input[_INPUT_NUM]   = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float *f32_in_buffer[_INPUT_NUM]   = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM] = {NULL};
    vsi_size_t   in_stride_size[_INPUT_NUM][VSI_NN_MAX_DIM_NUM]   = {{1}};
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM]    = {0};
    vsi_size_t i, b;
    int32_t  recurrent_activation = 0;
    vsi_size_t n_batch               = 0;
    vsi_size_t n_cell                = 0;

    /* prepare data */
    for( i = 0; i < _INPUT_NUM; i++ )
    {
        input[i]   = (vsi_nn_kernel_tensor_t)param[i];
        if (input[i])
        {
            in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
            vsi_nn_kernel_tensor_attr_get_stride( in_attr[i], in_stride_size[i] );
            f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
            CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );
        }
    }

    for( i = 0; i < _OUTPUT_NUM; i++ )
    {
        output[i]   = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        if (output[i])
        {
            out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );
            vsi_nn_kernel_tensor_attr_get_stride( out_attr[i], out_stride_size[i] );
            out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
            out_bytes[i] = out_elements[i] * sizeof(float);
            f32_out_buffer[i] = (float *)malloc( out_bytes[i] );
            CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
            memset( f32_out_buffer[i], 0, out_bytes[i] );
        }
    }

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_R_ACTIVATION],
        &recurrent_activation );
    CHECK_STATUS_FAIL_GOTO(status, final );
    n_cell  = in_attr[0]->shape->data[0];
    n_batch = in_attr[0]->shape->data[1];

    for (b = 0; b < n_batch; b ++)
    {
        for (i = 0; i < n_cell; i++)
        {
            vsi_size_t index = i + n_cell * b;
            float data_r_t = 0;
            float r_times_h = 0;
            float hstate_in = f32_in_buffer[0][index];

            data_r_t = f32_in_buffer[1][index];
            data_r_t += f32_in_buffer[2][index];

            data_r_t = vsi_nn_activation(data_r_t, recurrent_activation);

            r_times_h = hstate_in * data_r_t;

            f32_out_buffer[0][index] = r_times_h;
        }
    }

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        if (output[i])
        {
            status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                    f32_out_buffer[i], out_elements[i] );
            CHECK_STATUS_FAIL_GOTO( status, final );
        }
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
    vsi_status status = VSI_SUCCESS;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _grucell_h_times_activation_r_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _grucell_h_times_activation_r_kernel_param_def );

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
    vsi_nn_kernel_node_param_t node_params[_GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_R_ACTIVATION] = vsi_nn_kernel_scalar_create(
                    graph, I32, &recurrent_activation );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GRUCELL_H_TIMES_ACTIVATION_R_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_R_ACTIVATION] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( grucell_h_times_activation_r, _setup )

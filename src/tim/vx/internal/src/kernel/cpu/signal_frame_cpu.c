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
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.signal_frame")

/*
 * Kernel params
 */
static vx_param_description_t _signal_frame_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SIGNAL_FRAME_PARAM_NUM  _cnt_of_array( _signal_frame_kernel_param_def )
#define FRAME_LENGHT    (2)
#define FRAME_STEP      (3)
#define AXIS            (4)
#define PAD_END         (5)
#define PAD_VAL         (6)

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
    vsi_nn_kernel_tensor_t input[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float *f32_in_buffer[_INPUT_NUM] = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM] = {NULL};
    size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    size_t   out_elements[_OUTPUT_NUM] = {0};
    size_t   out_bytes[_OUTPUT_NUM] = {0};
    int32_t i = 0;
    int32_t j = 0;
    int32_t k = 0;
    int32_t frame_length = 0;
    int32_t frame_step = 0;
    int32_t axis = 0;
    int32_t pad_end = 0;
    int32_t length_samples = 0;
    int32_t num_frames = 0;
    int32_t inner_dim = 1;
    int32_t outer_dim = 1;
    int32_t inner_size = 1;
    float pad_val = 0;

    /* prepare data */
    for (i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input buffer fail.", final );
    }
    for (i = 0; i < _OUTPUT_NUM; i ++)
    {
        output[i] = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );
        vsi_nn_kernel_tensor_attr_get_stride( out_attr[i], out_stride_size[i] );
        out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
        out_bytes[i] = out_elements[i] * sizeof(float);
        f32_out_buffer[i] = (float *)malloc( out_bytes[i] );
        CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
        memset( f32_out_buffer[i], 0, out_bytes[i] );
    }

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[FRAME_LENGHT], &frame_length);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[FRAME_STEP], &frame_step);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[AXIS], &axis);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[PAD_END], &pad_end);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[PAD_VAL], &pad_val);
    CHECK_STATUS_FAIL_GOTO( status, final );

    for (i = 0; i < axis; i++)
    {
        inner_dim *= in_attr[0]->shape->data[i];
    }
    length_samples = in_attr[0]->shape->data[axis];
    for (i = axis + 1; i < (int32_t)in_attr[0]->shape->size; i++)
    {
        outer_dim *= in_attr[0]->shape->data[i];
    }

    for (i = 0; i < axis + 1; i++)
    {
        inner_size *= out_attr[0]->shape->data[i];
    }

    num_frames = (length_samples + frame_step - 1) / frame_step;
    num_frames = pad_end ? num_frames : (length_samples - frame_length) / frame_step + 1;

    for (i = 0; i < outer_dim; i++)
    {
        float * src_ptr = f32_in_buffer[0] + i * length_samples * inner_dim;
        float * dst_ptr = f32_out_buffer[0] + i * num_frames * frame_length * inner_dim;

        for (j = 0; j < num_frames; j++)
        {
            for (k = 0; k < frame_length; k++)
            {
                int32_t m = j * frame_step + k;

                if (pad_end)
                {
                    if (m >= length_samples)
                    {
                        int32_t l = 0;
                        for (l = 0; l < inner_dim; l++)
                        {
                            (dst_ptr + (j * frame_length + k) * inner_dim)[l] = pad_val;
                        }
                    }
                    else
                    {
                        memcpy(dst_ptr + (j * frame_length + k) * inner_dim, src_ptr + m * inner_dim,
                            inner_dim * sizeof(float));
                    }
                }
                else
                {
                    memcpy(dst_ptr + (j * frame_length + k) * inner_dim, src_ptr + m * inner_dim,
                        inner_dim * sizeof(float));
                }
            }
        }
    }

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                f32_out_buffer[i], out_elements[i] );
        CHECK_STATUS_FAIL_GOTO( status, final );
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
    for (i = 0; i < _OUTPUT_NUM; i++)
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
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _signal_frame_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _signal_frame_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_SIGNAL_FRAME_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t frame_length  = vsi_nn_kernel_param_get_int32( params, "frame_length" );
    int32_t frame_step  = vsi_nn_kernel_param_get_int32( params, "frame_step" );
    int32_t axis  = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t pad_end  = vsi_nn_kernel_param_get_int32( params, "pad_end" );
    float pad_val  = vsi_nn_kernel_param_get_float32( params, "pad_val" );

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SIGNAL_FRAME_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[FRAME_LENGHT] = vsi_nn_kernel_scalar_create( graph, I32, &frame_length );
            node_params[FRAME_STEP] = vsi_nn_kernel_scalar_create( graph, I32, &frame_step );
            node_params[AXIS] = vsi_nn_kernel_scalar_create( graph, I32, &axis );
            node_params[PAD_END] = vsi_nn_kernel_scalar_create( graph, I32, &pad_end );
            node_params[PAD_VAL] = vsi_nn_kernel_scalar_create( graph, F32, &pad_val );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SIGNAL_FRAME_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[FRAME_LENGHT] );
            vsi_nn_kernel_scalar_release( &node_params[FRAME_STEP] );
            vsi_nn_kernel_scalar_release( &node_params[AXIS] );
            vsi_nn_kernel_scalar_release( &node_params[PAD_END] );
            vsi_nn_kernel_scalar_release( &node_params[PAD_VAL] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( signal_frame, _setup )

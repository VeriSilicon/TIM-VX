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
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.resize_bilinear")


/*
 * Kernel params
 */
static vx_param_description_t _resize_bilinear_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _RESIZE_BILINEAR_PARAM_NUM  _cnt_of_array( _resize_bilinear_kernel_param_def )

#define SCALAR_ALIGN_CORNERS         (2)
#define SCALAR_HALF_PIXEL            (3)

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
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM];
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM];
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t i;
    int32_t  align_corners;
    int32_t  half_pixel_centers;
    float    width_scale;
    float    height_scale;
    vsi_size_t input_width, output_width, input_height, output_height;
    vsi_size_t b = 0, d = 0, w = 0, h = 0;
    vsi_size_t output_depth, input_depth;
    vsi_size_t output_batch;
    vsi_size_t output_dims, input_dims;
    float    data00 = .0f, data01 = .0f, data10 = .0f, data11 = .0f, interpolation = .0f;
    vsi_size_t input_width_orig;
    vsi_size_t output_width_orig;
    vsi_size_t index;

    /* prepare data */
    for(i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );
    }
    for(i = 0; i < _OUTPUT_NUM; i ++)
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

    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_ALIGN_CORNERS], &(align_corners));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_HALF_PIXEL], &(half_pixel_centers));
    input_width       = in_attr[0]->shape->data[0];
    input_height      = in_attr[0]->shape->data[1];
    output_width      = out_attr[0]->shape->data[0];
    output_height     = out_attr[0]->shape->data[1];
    output_dims       = (vsi_size_t)out_attr[0]->shape->size;
    output_depth      = output_dims > 2 ? out_attr[0]->shape->data[2] : 1;
    output_batch      = output_dims > 3 ? out_attr[0]->shape->data[3] : 1;
    input_dims        = (vsi_size_t)in_attr[0]->shape->size;
    input_depth       = input_dims > 2 ? in_attr[0]->shape->data[2] : 1;
    input_width_orig  = input_width;
    output_width_orig = output_width;

    if (align_corners && output_width > 1)
    {
        width_scale = ((vx_float32)(input_width - 1) * 1.0f) / (vx_float32)(output_width - 1);
    }
    else
    {
        width_scale = ((vx_float32)input_width * 1.0f) / (vx_float32)output_width;
    }

    if (align_corners && output_height > 1)
    {
        height_scale = ((vx_float32)(input_height - 1) * 1.0f) / (vx_float32)(output_height - 1);
    }
    else
    {
        height_scale = ((vx_float32)input_height * 1.0f) / (vx_float32)output_height;
    }

    for (b = 0; b < output_batch; b ++)
    {
        for (d = 0; d < output_depth; d ++)
        {
            vsi_ssize_t input_base = b * input_depth * input_width_orig * input_height \
            + d * input_width_orig * input_height;
            vsi_ssize_t output_base = b * output_depth * output_width_orig * output_height \
            + d * output_width_orig * output_height;

            for (h = 0; h < output_height; h ++)
            {
                vx_float32 input_h = h * height_scale;
                vsi_size_t h0;
                vsi_size_t h1;

                if (half_pixel_centers)
                {
                    input_h = ((vx_float32)h + 0.5f) * height_scale - 0.5f;
                }
                else
                {
                    input_h = h * height_scale;
                }
                h0 = (vsi_size_t)input_h;
                h1 = input_h < 0 ? 0 : vsi_nn_min(h0 + 1, input_height - 1);
                for (w = 0; w < output_width; w ++)
                {
                    vx_float32 input_w;
                    vsi_ssize_t w0;
                    vsi_ssize_t w1;
                    if (half_pixel_centers)
                    {
                        input_w = ((vx_float32)w + 0.5f) * width_scale - 0.5f;
                    }
                    else
                    {
                        input_w = w * width_scale;
                    }
                    w0 = (vsi_ssize_t)input_w;
                    w1 = input_w < 0 ? 0 : vsi_nn_min(w0 + 1, (vsi_ssize_t)(input_width - 1));
                    index = input_base + h0 * input_width_orig + w0;
                    data00 = f32_in_buffer[0][index];
                    index = input_base + h0 * input_width_orig + w1;
                    data01 = f32_in_buffer[0][index];
                    index = input_base + h1 * input_width_orig + w0;
                    data10 = f32_in_buffer[0][index];
                    index = input_base + h1 * input_width_orig + w1;
                    data11 = f32_in_buffer[0][index];

                    interpolation = data00 * (1 - (input_h - h0)) * (1 - (input_w - w0)) +
                                    data10 * (input_h - h0) * (1 - (input_w - w0)) +
                                    data01 * (1 - (input_h - h0)) * (input_w - w0) +
                                    data11 * (input_h - h0) * (input_w - w0);
                    index = output_base + h * output_width_orig + w;
                    f32_out_buffer[0][index] = interpolation;
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
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _resize_bilinear_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _resize_bilinear_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_RESIZE_BILINEAR_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );

    status = _query_kernel( kernel, inputs, outputs );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _RESIZE_BILINEAR_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_ALIGN_CORNERS] = vsi_nn_kernel_scalar_create( graph, I32, &align_corners );
            node_params[SCALAR_HALF_PIXEL] = vsi_nn_kernel_scalar_create( graph, I32, &half_pixel_centers );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _RESIZE_BILINEAR_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_ALIGN_CORNERS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_HALF_PIXEL] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( resize_bilinear, _setup )

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
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.upsample")


/*
 * Kernel params
 */
static vx_param_description_t _upsample_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _UPSAMPLE_PARAM_NUM  _cnt_of_array( _upsample_kernel_param_def )

#define SCALAR_KSZIE_X          (3)
#define SCALAR_KSZIE_Y          (4)

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
    int32_t  i, j, b, p;
    int32_t  batch, depth, height, width, height_o, width_o;
    int32_t  input_base  = 0;
    int32_t  output_base = 0;
    int32_t  ksize_x     = 0;
    int32_t  ksize_y     = 0;
    vsi_bool is_relative_coord = FALSE;
    vsi_nn_kernel_dtype_e input1_dtype;

    for(i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        if (1 == i)
        {
            in_attr[1]->quant = VSI_NN_KERNEL_QUANT_NONE;
        }
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

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_KSZIE_X],  &ksize_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_KSZIE_Y],  &ksize_y);

    batch    = in_attr[0]->shape->size > 3 ? in_attr[0]->shape->data[3] : 1;
    depth    = in_attr[0]->shape->size > 2 ? in_attr[0]->shape->data[2] : 1;
    width    = in_attr[0]->shape->data[0];
    height   = in_attr[0]->shape->data[1];
    width_o  = out_attr[0]->shape->data[0];
    height_o = out_attr[0]->shape->data[1];
    input1_dtype = in_attr[1]->dtype;

    if ((I8 == input1_dtype) || (U8 == input1_dtype) || (I16 == input1_dtype))
    {
        is_relative_coord = TRUE;
    }


    for(b = 0; b < batch; b++)
    {
        for (p = 0; p < depth; p ++)
        {
            input_base  = b * depth * height * width + p * height * width;
            output_base = b * depth * height_o * width_o + p * height_o * width_o;
            for (j = 0; j < height; j ++)
            {
                for (i = 0; i < width; i ++)
                {
                    int32_t in_index  = input_base + j * width + i;
                    float   in_value  = f32_in_buffer[0][in_index];
                    int32_t up_index  = (int32_t)f32_in_buffer[1][in_index];
                    int32_t out_index = up_index;
                    if (is_relative_coord)
                    {
                        int32_t relative_y = up_index / ksize_x;
                        int32_t relative_x = up_index % ksize_x;
                        out_index = output_base + ((j * ksize_y) + relative_y) * width_o + i * ksize_x + relative_x;
                    }
                    f32_out_buffer[0][out_index] = in_value;
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
    kernel->info.parameters  = _upsample_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _upsample_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_UPSAMPLE_PARAM_NUM] = {NULL};
    int32_t  scale_x  = 0;
    int32_t  scale_y  = 0;
    vsi_nn_kernel_node_t node = NULL;

    scale_x  = vsi_nn_kernel_param_get_int32(params, "scale_x");
    scale_y  = vsi_nn_kernel_param_get_int32(params, "scale_y");

    status = _query_kernel( kernel, inputs, outputs );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _UPSAMPLE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_KSZIE_X] = vsi_nn_kernel_scalar_create(
                    graph, I32, &scale_x );
            node_params[SCALAR_KSZIE_Y] = vsi_nn_kernel_scalar_create(
                    graph, I32, &scale_y );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _UPSAMPLE_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_KSZIE_X] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_KSZIE_Y] );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( upsample, _setup )


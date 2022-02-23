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
#define _CPU_IO_NUM         (_INPUT_NUM + _OUTPUT_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.custom_warp_affine")


/*
 * Kernel params
 */
static vx_param_description_t _custom_warp_affine_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CUSTOM_WARP_AFFINE_PARAM_NUM  _cnt_of_array( _custom_warp_affine_kernel_param_def )
#define SCALAR_INPUT_TYPE       (2)
#define SCALAR_MATRIX_OFFSET    (3)

static void _transform_affine
    (
    vsi_size_t dst_x,
    vsi_size_t dst_y,
    const float m[],
    float *src_x,
    float *src_y
    )
{
    *src_x = dst_x * m[0] + dst_y * m[2] + m[4];
    *src_y = dst_x * m[1] + dst_y * m[3] + m[5];
}

static vsi_bool _read_pixel
    (
    float *base,
    vsi_nn_kernel_tensor_attr_t *attr,
    float x,
    float y,
    float *pixel
    )
{
    vsi_size_t width = attr->shape->data[0];
    vsi_size_t height = attr->shape->data[1];
    vsi_bool out_of_bounds = (x < 0 || y < 0 || x >= width || y >= height);
    vsi_size_t bx = 0, by = 0;

    if (out_of_bounds)
    {
        *pixel = 205.0f;
        return TRUE;
    }

    // bounded x/y
    bx = x < 0 ? 0 : x >= width ? width - 1 : (vsi_size_t)x;
    by = y < 0 ? 0 : y >= height ? height - 1 : (vsi_size_t)y;

    *pixel = base[by * width + bx];

    return TRUE;
}

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
    float* buffer[_CPU_IO_NUM] = { NULL };
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    vsi_nn_kernel_tensor_attr_t* attr[_CPU_IO_NUM] = { NULL };
    int32_t type = 0;
    float matrix[6] = {0};
    vsi_size_t i = 0;
    vsi_size_t b = 0;
    vsi_size_t x = 0;
    vsi_size_t y = 0;
    vsi_size_t out_elements = 0;
    vsi_size_t width = 0;
    vsi_size_t height = 0;
    vsi_size_t outer_size = 1;

    tensors[0] = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1] = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    /* alloc the float32 data buffer */
    buffer[1] = (float *)malloc(out_elements * sizeof(float));
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input buffer fail.", final );
    memset(buffer[1], 0, out_elements * sizeof(float));

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_TYPE],
        &type);
    CHECK_STATUS_FAIL_GOTO(status, final );
    for (i = 0; i < 6; i++)
    {
        status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_MATRIX_OFFSET + i],
            &matrix[i]);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    width = attr[1]->shape->data[0];
    height = attr[1]->shape->data[1];
    for(i = 2; i < (vsi_size_t)attr[1]->shape->size; ++i)
    {
        outer_size *= attr[1]->shape->data[i];
    }
    // Do something
    for (b = 0; b < outer_size; b++)
    {
        float *src_base = buffer[0] + b * attr[0]->shape->data[0] * attr[0]->shape->data[1];
        float *dst_base = buffer[1] + b * width * height;
        for (y = 0; y < height; y++)
        {
            for (x = 0; x < width; x++)
            {
                float xf = 0;
                float yf = 0;
                float dst = 0;

                _transform_affine(x, y, matrix, &xf, &yf);
                if (type == VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR)
                {
                    _read_pixel(src_base, attr[0], xf, yf, &dst);
                    dst_base[y * width + x] = dst;
                }
                else
                {
                    float tl = 0, tr = 0, bl = 0, br = 0;
                    float ar = xf - floorf(xf);
                    float ab = yf - floorf(yf);
                    float al = 1.0f - ar;
                    float at = 1.0f - ab;

                    _read_pixel(src_base, attr[0], floorf(xf), floorf(yf), &tl);
                    _read_pixel(src_base, attr[0], floorf(xf) + 1, floorf(yf), &tr);
                    _read_pixel(src_base, attr[0], floorf(xf), floorf(yf) + 1, &bl);
                    _read_pixel(src_base, attr[0], floorf(xf) + 1, floorf(yf) + 1, &br);

                    dst_base[y * width + x] = tl * al * at + tr * ar * at + bl * al * ab + br * ar * ab;
                }
            }
        }
    }

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
    kernel->info.parameters  = _custom_warp_affine_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _custom_warp_affine_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_CUSTOM_WARP_AFFINE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    size_t i = 0;
    size_t buffer_size = 0;
    int32_t type = vsi_nn_kernel_param_get_int32( params, "type");
    float * buffer = (float*)vsi_nn_kernel_param_get_const_buffer( params, "matrix", &buffer_size );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CUSTOM_WARP_AFFINE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_INPUT_TYPE] = vsi_nn_kernel_scalar_create(
                graph, I32, &type );
            for (i = 0; i < buffer_size; i++)
            {
                node_params[SCALAR_MATRIX_OFFSET + i] = vsi_nn_kernel_scalar_create(
                        graph, F32, &buffer[i] );
            }

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CUSTOM_WARP_AFFINE_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TYPE] );
            for (i = 0; i < buffer_size; i++)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_MATRIX_OFFSET + i] );
            }
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( custom_warp_affine, _setup )

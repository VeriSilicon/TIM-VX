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
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.spatial_transformer")


/*
 * Kernel params
 */
static vx_param_description_t _spatial_transformer_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _SPATIAL_TRANSFORMER_PARAM_NUM  _cnt_of_array( _spatial_transformer_kernel_param_def )
#define HAS_THETA_1_1   (3)
#define HAS_THETA_1_2   (4)
#define HAS_THETA_1_3   (5)
#define HAS_THETA_2_1   (6)
#define HAS_THETA_2_2   (7)
#define HAS_THETA_2_3   (8)
#define THETA_1_1       (9)
#define THETA_1_2       (10)
#define THETA_1_3       (11)
#define THETA_2_1       (12)
#define THETA_2_2       (13)
#define THETA_2_3       (14)
#define ALIGN_CORNERS   (15)

static void _transform_affine(int32_t dst_x, int32_t dst_y, const float m[], float *src_x, float *src_y)
{
    *src_x = dst_x * m[0] + dst_y * m[2] + m[4];
    *src_y = dst_x * m[1] + dst_y * m[3] + m[5];
}

static float _read_pixel(float *base, vsi_nn_kernel_tensor_attr_t *attr,
                          float x, float y, int32_t z, int32_t b)
{
    vsi_bool out_of_bounds = (x < 0 || y < 0 || x >= attr->shape->data[0] || y >= attr->shape->data[1]);
    int32_t bx, by;
    int32_t offset = (b * attr->shape->data[2] + z) * attr->shape->data[0] * attr->shape->data[1];
    float pixel = 0;

    if (out_of_bounds)
    {
        return 0;
    }
    // bounded x/y
    bx = (int32_t)x;
    by = (int32_t)y;

    pixel = base[attr->shape->data[0] * by + bx + offset];

    return pixel;
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
    vsi_nn_kernel_tensor_t input[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float *f32_in_buffer[_INPUT_NUM] = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM] = {NULL};
    size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    size_t   out_elements[_OUTPUT_NUM] = {0};
    size_t   out_bytes[_OUTPUT_NUM] = {0};
    int32_t  i = 0;
    int32_t  b = 0;
    int32_t  c = 0;
    int32_t  j = 0;
    int32_t  x = 0;
    int32_t  y = 0;
    int32_t  has_theta[6] = {0};
    int32_t  batch = 1;
    int32_t  depth = 1;
    int32_t  height = 1;
    int32_t  width = 1;
    int32_t  input_height = 1;
    int32_t  input_width = 1;
    int32_t  rank = 0;
    int32_t  index = 0;
    int32_t  align_corners = 0;
    float    theta[6] = {0};

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

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[HAS_THETA_1_1], &has_theta[0]);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[HAS_THETA_1_2], &has_theta[1]);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[HAS_THETA_1_3], &has_theta[2]);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[HAS_THETA_2_1], &has_theta[3]);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[HAS_THETA_2_2], &has_theta[4]);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[HAS_THETA_2_3], &has_theta[5]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_1_1], &theta[0]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_1_2], &theta[1]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_1_3], &theta[2]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_2_1], &theta[3]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_2_2], &theta[4]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_2_3], &theta[5]);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[ALIGN_CORNERS], &align_corners);
    CHECK_STATUS_FAIL_GOTO( status, final );

    rank = (int32_t)out_attr[0]->shape->size;
    width = out_attr[0]->shape->data[0];
    height = out_attr[0]->shape->data[1];
    depth = rank > 2 ? out_attr[0]->shape->data[2] : 1;
    batch = rank > 3 ? out_attr[0]->shape->data[3] : 1;

    input_width = in_attr[0]->shape->data[0];
    input_height = in_attr[0]->shape->data[1];

    for (b = 0; b < batch; b++)
    {
        float _w = (float)input_width;
        float _h = (float)input_height;
        float w = (float)width;
        float h = (float)height;
        float matrix_m[6] = {0};
        j = 0;
        for (i = 0; i < 6; i++)
        {
            if (has_theta[i] == 0)
            {
                theta[i] = f32_in_buffer[1][b * in_attr[1]->shape->data[0] + j];
                j ++;
            }
        }

        if (align_corners && w > 1)
        {
            w = w - 1;
        }

        if (align_corners && h > 1)
        {
            h = h - 1;
        }

        matrix_m[0] = theta[4] * _w / w;
        matrix_m[2] = theta[3] * _w / h;
        matrix_m[4] = (theta[5] - theta[4] - theta[3] + 1) * _w * 0.5f;
        matrix_m[1] = theta[1] * _h / w;
        matrix_m[3] = theta[0] * _h / h;
        matrix_m[5] = (theta[2] - theta[1] - theta[0] + 1) * _h * 0.5f;
        for (c = 0; c < depth; c++)
        {
            for (y = 0; y < height; y++)
            {
                for (x = 0; x < width; x++)
                {
                    float xf = 0;
                    float yf = 0;
                    float tl = 0, tr = 0, bl = 0, br = 0;
                    float ar = 0, ab = 0, al = 0, at = 0;

                    _transform_affine(x, y, matrix_m, &xf, &yf);

                    xf = xf < 0 ? xf - 1 : xf;
                    yf = yf < 0 ? yf - 1 : yf;
                    ar = xf - floorf(xf);
                    ab = yf - floorf(yf);
                    al = 1.0f - ar;
                    at = 1.0f - ab;

                    tl = _read_pixel(f32_in_buffer[0], in_attr[0], floorf(xf), floorf(yf), c, b);
                    tr = _read_pixel(f32_in_buffer[0], in_attr[0], floorf(xf) + 1, floorf(yf), c, b);
                    bl = _read_pixel(f32_in_buffer[0], in_attr[0], floorf(xf), floorf(yf) + 1, c, b);
                    br = _read_pixel(f32_in_buffer[0], in_attr[0], floorf(xf) + 1, floorf(yf) + 1, c, b);

                    f32_out_buffer[0][index ++] = tl * al * at + tr * ar * at + bl * al * ab + br * ar * ab;
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
    /* Add extra params */
    )
{
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _spatial_transformer_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _spatial_transformer_kernel_param_def );

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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_SPATIAL_TRANSFORMER_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t has_theta_1_1  = vsi_nn_kernel_param_get_int32( params, "has_theta_1_1" );
    int32_t has_theta_1_2  = vsi_nn_kernel_param_get_int32( params, "has_theta_1_2" );
    int32_t has_theta_1_3  = vsi_nn_kernel_param_get_int32( params, "has_theta_1_3" );
    int32_t has_theta_2_1  = vsi_nn_kernel_param_get_int32( params, "has_theta_2_1" );
    int32_t has_theta_2_2  = vsi_nn_kernel_param_get_int32( params, "has_theta_2_2" );
    int32_t has_theta_2_3  = vsi_nn_kernel_param_get_int32( params, "has_theta_2_3" );
    float theta_1_1  = vsi_nn_kernel_param_get_float32( params, "theta_1_1" );
    float theta_1_2  = vsi_nn_kernel_param_get_float32( params, "theta_1_2" );
    float theta_1_3  = vsi_nn_kernel_param_get_float32( params, "theta_1_3" );
    float theta_2_1  = vsi_nn_kernel_param_get_float32( params, "theta_2_1" );
    float theta_2_2  = vsi_nn_kernel_param_get_float32( params, "theta_2_2" );
    float theta_2_3  = vsi_nn_kernel_param_get_float32( params, "theta_2_3" );
    int32_t align_corners  = vsi_nn_kernel_param_get_int32( params, "align_corners" );

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SPATIAL_TRANSFORMER_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[HAS_THETA_1_1] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_1_1 );
            node_params[HAS_THETA_1_2] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_1_2 );
            node_params[HAS_THETA_1_3] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_1_3 );
            node_params[HAS_THETA_2_1] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_2_1 );
            node_params[HAS_THETA_2_2] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_2_2 );
            node_params[HAS_THETA_2_3] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_2_3 );
            node_params[THETA_1_1] = vsi_nn_kernel_scalar_create( graph, F32, &theta_1_1 );
            node_params[THETA_1_2] = vsi_nn_kernel_scalar_create( graph, F32, &theta_1_2 );
            node_params[THETA_1_3] = vsi_nn_kernel_scalar_create( graph, F32, &theta_1_3 );
            node_params[THETA_2_1] = vsi_nn_kernel_scalar_create( graph, F32, &theta_2_1 );
            node_params[THETA_2_2] = vsi_nn_kernel_scalar_create( graph, F32, &theta_2_2 );
            node_params[THETA_2_3] = vsi_nn_kernel_scalar_create( graph, F32, &theta_2_3 );
            node_params[ALIGN_CORNERS] = vsi_nn_kernel_scalar_create( graph, I32, &align_corners );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SPATIAL_TRANSFORMER_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_1_1] );
            vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_1_2] );
            vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_1_3] );
            vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_2_1] );
            vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_2_2] );
            vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_2_3] );
            vsi_nn_kernel_scalar_release( &node_params[THETA_1_1] );
            vsi_nn_kernel_scalar_release( &node_params[THETA_1_2] );
            vsi_nn_kernel_scalar_release( &node_params[THETA_1_3] );
            vsi_nn_kernel_scalar_release( &node_params[THETA_2_1] );
            vsi_nn_kernel_scalar_release( &node_params[THETA_2_2] );
            vsi_nn_kernel_scalar_release( &node_params[THETA_2_3] );
            vsi_nn_kernel_scalar_release( &node_params[ALIGN_CORNERS] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( spatial_transformer, _setup )

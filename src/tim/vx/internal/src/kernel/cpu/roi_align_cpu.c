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
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.roi_align")


/*
 * Kernel params
 */
static vx_param_description_t _roi_align_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _ROI_ALIGN_PARAM_NUM  _cnt_of_array( _roi_align_kernel_param_def )
#define SCALAR_X_RATIO          (4)
#define SCALAR_Y_RATIO          (5)
#define SCALAR_X_SAMPLE         (6)
#define SCALAR_Y_SAMPLE         (7)

/*
 * Kernel function
 */
static float _compute_region_coordinate(int32_t p, float bin_size, float roi_anchor, float max_value)
{
    const float region_start = p * bin_size + roi_anchor;

    return region_start;
}

static float _roi_align_1x1(float *input_ptr,
                           int32_t width,
                           int32_t height,
                           float   region_start_x,
                           float   bin_size_x,
                           int32_t grid_size_x,
                           float   region_end_x,
                           float   region_start_y,
                           float   bin_size_y,
                           int32_t grid_size_y,
                           float   region_end_y)
{
    float avg = 0;
    int32_t iy = 0;
    int32_t ix = 0;
    // Iterate through the aligned pooling region
    for (iy = 0; iy < grid_size_y; ++iy)
    {
        for (ix = 0; ix < grid_size_x; ++ix)
        {
            // Align the window in the middle of every bin
            float y = region_start_y +
                      ((float)iy + 0.5f) * bin_size_y / (float)(grid_size_y);
            float x = region_start_x +
                      ((float)ix + 0.5f) * bin_size_x / (float)(grid_size_x);

            // Interpolation in the [0,0] [0,1] [1,0] [1,1] square
            const int32_t y_low = vsi_nn_min((int32_t)y, height - 1);
            const int32_t x_low = vsi_nn_min((int32_t)x, width - 1);
            const int32_t y_high = vsi_nn_min(y_low + 1, height - 1);
            const int32_t x_high = vsi_nn_min(x_low + 1, width - 1);

            float ly = y - y_low;
            float lx = x - x_low;
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;

            float w1 = hy * hx;
            float w2 = hy * lx;
            float w3 = ly * hx;
            float w4 = ly * lx;

            const float data1 = *(input_ptr + y_low * width + x_low);
            const float data2 = *(input_ptr + y_low * width + x_high);
            const float data3 = *(input_ptr + y_high * width + x_low);
            const float data4 = *(input_ptr + y_high * width + x_high);

            /* onnx: inverse elements are out of feature map boundary */
            if (x > width || x < -1 || y > height || y < -1) continue;

            x = x_low >= width - 1 ? x_low : x;
            y = y_low >= height - 1 ? y_low : y;

            ly = y - y_low;
            lx = x - x_low;
            hy = 1.0f - ly;
            hx = 1.0f - lx;

            w1 = hy * hx;
            w2 = hy * lx;
            w3 = ly * hx;
            w4 = ly * lx;

            avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
        }
    }

    avg /= grid_size_x * grid_size_y;

    return avg;
}

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
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i                 = 0;
    float     width_scale       = 0.0f;
    float     height_scale      = 0.0f;
    float     width_ratio       = 0.0f;
    float     height_ratio      = 0.0f;
    int32_t   width_sample_num  = 0;
    int32_t   height_sample_num = 0;
    uint32_t  n                 = 0;
    vsi_size_t  num_rois          = 0;
    vsi_ssize_t   inHeight          = 0;
    vsi_ssize_t   inWidth           = 0;
    vsi_ssize_t   inDepth           = 0;
    vsi_ssize_t   outHeight         = 0;
    vsi_ssize_t   outWidth          = 0;
    uint32_t  kRoiDim           = 4;
    uint32_t  out_index         = 0;

    /* prepare data */
    for (i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );
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

    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_X_RATIO], &(width_ratio));
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_Y_RATIO], &(height_ratio));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_X_SAMPLE], &(width_sample_num));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_Y_SAMPLE], &(height_sample_num));

    width_scale = 1.0f / width_ratio;
    height_scale = 1.0f / height_ratio;
    num_rois = in_attr[1]->shape->data[1];

    inWidth = in_attr[0]->shape->data[0];
    inHeight = in_attr[0]->shape->data[1];
    inDepth = in_attr[0]->shape->data[2];
    outWidth = out_attr[0]->shape->data[0];
    outHeight = out_attr[0]->shape->data[1];

    for (n = 0; n < num_rois; n++)
    {
        uint32_t batchId = (uint32_t)f32_in_buffer[2][n];
        float qx1 = f32_in_buffer[1][n * kRoiDim];
        float qy1 = f32_in_buffer[1][n * kRoiDim + 1];
        float qx2 = f32_in_buffer[1][n * kRoiDim + 2];
        float qy2 = f32_in_buffer[1][n * kRoiDim + 3];

        float x1 = qx1;
        float x2 = qx2;
        float y1 = qy1;
        float y2 = qy2;
        float roi_anchor_x = x1 * width_scale;
        float roi_anchor_y = y1 * height_scale;
        float roi_dims_x   = vsi_nn_max((x2 - x1) * width_scale, 1.0f);
        float roi_dims_y   = vsi_nn_max((y2 - y1) * height_scale, 1.0f);
        float bin_size_x   = roi_dims_x / outWidth;
        float bin_size_y   = roi_dims_y / outHeight;

        vsi_ssize_t batch_base_index = batchId * inHeight * inWidth * inDepth;
        int32_t ch = 0;
        int32_t py = 0;
        int32_t px = 0;

        for (ch = 0; ch < inDepth; ch++)
        {
            for (py = 0; py < outHeight; py++)
            {
                for (px = 0; px < outWidth; px++)
                {
                    float region_start_x = _compute_region_coordinate(px, bin_size_x,
                        roi_anchor_x, (float)inWidth);
                    float region_start_y = _compute_region_coordinate(py, bin_size_y,
                        roi_anchor_y, (float)inHeight);
                    float region_end_x   = _compute_region_coordinate(px + 1, bin_size_x,
                        roi_anchor_x, (float)inWidth);
                    float region_end_y   = _compute_region_coordinate(py + 1, bin_size_y,
                        roi_anchor_y, (float)inHeight);

                    int32_t roi_bin_grid_x = (width_sample_num > 0) ? width_sample_num : (int32_t)(ceil(bin_size_x));
                    int32_t roi_bin_grid_y = (height_sample_num > 0) ? height_sample_num : (int32_t)(ceil(bin_size_y));

                    float *input_ptr = &f32_in_buffer[0][batch_base_index + ch * inWidth * inHeight];
                    float out_val = 0;

                    out_val = _roi_align_1x1(
                        input_ptr, (int32_t)inWidth, (int32_t)inHeight, region_start_x, bin_size_x,
                        roi_bin_grid_x, region_end_x, region_start_y, bin_size_y,
                        roi_bin_grid_y, region_end_y);

                    f32_out_buffer[0][out_index++] = out_val;
                }
            }
        }
    }

    /* save data */
    for (i = 0; i < _OUTPUT_NUM; i++)
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
    )
{
    vsi_status status = VSI_FAILURE;

    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _roi_align_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _roi_align_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_ROI_ALIGN_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float   width_ratio         = vsi_nn_kernel_param_get_float32( params, "width_ratio" );
    float   height_ratio        = vsi_nn_kernel_param_get_float32( params, "height_ratio" );
    int32_t width_sample_num    = vsi_nn_kernel_param_get_int32( params, "width_sample_num" );
    int32_t height_sample_num   = vsi_nn_kernel_param_get_int32( params, "height_sample_num" );

    status = _query_kernel( kernel, inputs, outputs );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ROI_ALIGN_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_X_RATIO] = vsi_nn_kernel_scalar_create( graph, F32, &width_ratio );
            node_params[SCALAR_Y_RATIO] = vsi_nn_kernel_scalar_create( graph, F32, &height_ratio );
            node_params[SCALAR_X_SAMPLE] = vsi_nn_kernel_scalar_create( graph, I32, &width_sample_num );
            node_params[SCALAR_Y_SAMPLE] = vsi_nn_kernel_scalar_create( graph, I32, &height_sample_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ROI_ALIGN_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_X_RATIO] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_Y_RATIO] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_X_SAMPLE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_Y_SAMPLE] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( roi_align, _setup )

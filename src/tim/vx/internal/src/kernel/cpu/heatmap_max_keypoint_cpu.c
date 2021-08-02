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
#include <float.h>
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
#define _OUTPUT_NUM         (2)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.heatmap_max_keypoint")


/*
 * Kernel params
 */
static vx_param_description_t _heatmap_max_keypoint_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _HEATMAP_MAX_KEYPOINT_PARAM_NUM  _cnt_of_array( _heatmap_max_keypoint_kernel_param_def )

// This function uses Taylor expansion up to the quatratic term to approximate bicubic
// upscaling result.
// 2nd order Taylor expansion: D(x) = D - b'x + 1/2 * x'Ax
// where D = grid[1][1], Taylor expansion center, the original score,
//       x = delta, the correction on max keypoint position,
//       D(x) = deltaScore, the accuracy score after correction
static void _solve_for_delta
    (
    const float grid[3][3],
    float* delta,
    float* deltaScore,
    float fpAtol,
    float fpRtol
    )
{
    // b: negative 1st order derivative at center
    // A: Hessian matrix at center (2nd order derivative)
    float A[2][2], b[2];
    float crossProd1, crossProd2;
    float detA;
    b[0] = -(grid[1][2] - grid[1][0]) / 2.0f;
    b[1] = -(grid[2][1] - grid[0][1]) / 2.0f;
    A[0][0] = grid[1][0] - 2.0f * grid[1][1] + grid[1][2];
    A[0][1] = (grid[2][2] - grid[2][0] - grid[0][2] + grid[0][0]) / 4.0f;
    A[1][0] = A[0][1];
    A[1][1] = grid[0][1] - 2.0f * grid[1][1] + grid[2][1];

    // solve Ax=b, where x=delta -> delta = inv(A) * b
    crossProd1 = A[0][0] * A[1][1];
    crossProd2 = A[0][1] * A[1][0];
    detA = crossProd1 - crossProd2;
    // check if A is invertible
    if (fabs(detA) < (fpAtol + fpRtol * crossProd1)) return;
    delta[0] = (A[1][1] * b[0] - A[0][1] * b[1]) / detA;
    delta[1] = (A[0][0] * b[1] - A[1][0] * b[0]) / detA;

    // clip out of range delta, i.e. delta > 3/2
    if (fabs(delta[0]) > 1.5f || fabs(delta[1]) > 1.5f)
    {
        float scale = (float)(1.5f / vsi_nn_max(fabs(delta[0]), fabs(delta[1])));
        delta[0] *= scale;
        delta[1] *= scale;
    }

    *deltaScore = grid[1][1] - b[0] * delta[0] - b[1] * delta[1] +
                  ((A[0][0] * delta[0] + A[0][1] * delta[1]) * delta[0] +
                   (A[1][0] * delta[0] + A[1][1] * delta[1]) * delta[1]) /
                          2.0f;
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
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM];
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM];
    size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    size_t   out_elements[_OUTPUT_NUM] = {0};
    size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i = 0;
    uint32_t j = 0;
    uint32_t k = 0;
    uint32_t numBoxes = 0;
    uint32_t heatmapSize = 0;
    uint32_t numKeypoints = 0;
    uint32_t boxInfoLength = 4;
    uint32_t output_score_index = 0;
    uint32_t output_keypoint_index = 0;

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

    numBoxes = in_attr[0]->shape->data[3];
    heatmapSize = in_attr[0]->shape->data[2];
    numKeypoints = in_attr[0]->shape->data[0];

    for(i = 0; i < numBoxes; i++)
    {
        for (j = 0; j < numKeypoints; j++)
        {
            uint32_t maxIndex = 0;
            float maxScore = -FLT_MAX;
            uint32_t maxIndexWidth;
            uint32_t maxIndexHeight;
            float localGrid[3][3] = {{0}};
            int32_t dh, dw;
            float delta[2] = {0.0f, 0.0f}, deltaScore;
            float wRoiStart = f32_in_buffer[1][i * boxInfoLength];
            float hRoiStart = f32_in_buffer[1][i * boxInfoLength + 1];
            float wRoiEnd = f32_in_buffer[1][i * boxInfoLength + 2];
            float hRoiEnd = f32_in_buffer[1][i * boxInfoLength + 3];
            float roiWidth = wRoiEnd - wRoiStart;
            float roiHeight = hRoiEnd - hRoiStart;
            float wRelativePos;
            float hRelativePos;
            for (k = 0; k < heatmapSize * heatmapSize; k++)
            {
                uint32_t index = i * heatmapSize * heatmapSize * numKeypoints
                    + k * numKeypoints + j;
                float val = f32_in_buffer[0][index];
                if (maxScore < val)
                {
                    maxScore = val;
                    maxIndex = k;
                }
            }
            maxIndexWidth = maxIndex % heatmapSize;
            maxIndexHeight = maxIndex / heatmapSize;

            // get local 3x3 grid
            for (dh = -1; dh <= 1; dh++)
            {
                for (dw = -1; dw <= 1; dw++)
                {
                    // cast uint32_t to int32_t
                    int32_t h = (int32_t)(maxIndexHeight) + dh;
                    int32_t w = (int32_t)(maxIndexWidth) + dw;
                    uint32_t heatmapIndex;

                    // use mirroring for out of bound indexing
                    // need to ensure heatmapSize >= 2
                    h = h < 0 ? 1 : (h >= (int32_t)heatmapSize ? heatmapSize - 2 : h);
                    w = w < 0 ? 1 : (w >= (int32_t)heatmapSize ? heatmapSize - 2 : w);

                    heatmapIndex = i * heatmapSize * heatmapSize * numKeypoints +
                        (uint32_t)(h) * heatmapSize * numKeypoints +
                        (uint32_t)(w) * numKeypoints + j;
                    localGrid[dh + 1][dw + 1] = f32_in_buffer[0][heatmapIndex];
                }
            }
            deltaScore = maxScore;
            _solve_for_delta((const float (*)[3])localGrid, delta, &deltaScore, 1e-3f, 1e-3f);

            wRelativePos = ((float)(maxIndexWidth) + delta[0] + 0.5f) /
                (float)(heatmapSize);
            hRelativePos = ((float)(maxIndexHeight) + delta[1] + 0.5f) /
                (float)(heatmapSize);
            f32_out_buffer[0][output_score_index] = deltaScore;
            f32_out_buffer[1][output_keypoint_index] = wRelativePos * roiWidth + wRoiStart;
            f32_out_buffer[1][output_keypoint_index + 1] = hRelativePos * roiHeight + hRoiStart;
            output_score_index++;
            output_keypoint_index += 2;
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
        vsi_nn_safe_free(f32_in_buffer[i]);

        if (in_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &in_attr[i] );
        }
    }
    for (i = 0; i < _OUTPUT_NUM; i++)
    {
        vsi_nn_safe_free(f32_out_buffer[i]);

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
    kernel->info.parameters  = _heatmap_max_keypoint_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _heatmap_max_keypoint_kernel_param_def );

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
    vsi_nn_kernel_node_param_t node_params[_HEATMAP_MAX_KEYPOINT_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _HEATMAP_MAX_KEYPOINT_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _HEATMAP_MAX_KEYPOINT_PARAM_NUM );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( heatmap_max_keypoint, _setup )

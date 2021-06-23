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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_HEATMAP_MAX_KEYPOINT)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_HEATMAP_MAX_KEYPOINT)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_HEATMAP_MAX_KEYPOINT)
#define _VX_KERNEL_FUNC_KERNEL  (vxHeatmap_max_keypointKernel)

// This function uses Taylor expansion up to the quatratic term to approximate bicubic
// upscaling result.
// 2nd order Taylor expansion: D(x) = D - b'x + 1/2 * x'Ax
// where D = grid[1][1], Taylor expansion center, the original score,
//       x = delta, the correction on max keypoint position,
//       D(x) = deltaScore, the accuracy score after correction
static void solveForDelta
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

static vsi_status VX_CALLBACK vxHeatmap_max_keypointKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (1)
#define TENSOR_NUM_INPUT (2)
#define TENSOR_NUM_OUTPUT (2)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t type;

    uint32_t i = 0;
    for(i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        memset(&in_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        memset(&out_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    /* prepare data */
    context = vxGetContext((vx_reference)node);

    for(i = 0; i < TENSOR_NUM_INPUT; i ++)
    {
        input[i] = (vx_tensor)paramObj[i];
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        TEST_CHECK_STATUS(status, final);
        in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        f32_in_buffer[i] = (float *)malloc(in_elements[i] * sizeof(float));
        status = vsi_nn_vxConvertTensorToFloat32Data(
            context, input[i], &in_attr[i], f32_in_buffer[i],
            in_elements[i] * sizeof(float));
        TEST_CHECK_STATUS(status, final);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        f32_out_buffer[i]= (float *)malloc(out_elements[i] * sizeof(float));
        memset(f32_out_buffer[i], 0, out_elements[i] * sizeof(float));
    }
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(type),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    {
        uint32_t j, k;
        uint32_t numBoxes = in_attr[0].size[3];
        uint32_t heatmapSize = in_attr[0].size[2];
        uint32_t numKeypoints = in_attr[0].size[0];
        uint32_t boxInfoLength = 4;
        uint32_t output_score_index = 0;
        uint32_t output_keypoint_index = 0;

        for(i = 0; i < numBoxes; i++)
        {
            for (j = 0; j < numKeypoints; j++)
            {
                uint32_t maxIndex = 0;
                float maxScore = -FLT_MAX;
                uint32_t maxIndexWidth;
                uint32_t maxIndexHeight;
                float localGrid[3][3];
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
                solveForDelta((const float (*)[3])localGrid, delta, &deltaScore, 1e-3f, 1e-3f);

                wRelativePos = ((float)(maxIndexWidth) + delta[0] + 0.5f) /
                    (float)(heatmapSize);
                hRelativePos = ((float)(maxIndexHeight) + delta[1] + 0.5f) /
                    (float)(heatmapSize);
                f32_out_buffer[0][output_score_index] = deltaScore;
                f32_out_buffer[1][output_keypoint_index] = wRelativePos * roiWidth + wRoiStart;
                f32_out_buffer[1][output_keypoint_index + 1] = hRelativePos * roiHeight + hRoiStart;
                output_score_index++;
                output_keypoint_index +=2;
            }
        }
    }

    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        status = vsi_nn_vxConvertFloat32DataToTensor(
            context, output[i], &out_attr[i], f32_out_buffer[i],
            out_elements[i] * sizeof(float));
        TEST_CHECK_STATUS(status, final);
    }

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i]) free(f32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxHeatmap_max_keypointKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxHeatmap_max_keypointInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxHeatmap_max_keypoint_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxHeatmap_max_keypointKernelParam,
    _cnt_of_array( vxHeatmap_max_keypointKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxHeatmap_max_keypoint_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxHeatmap_max_keypointKernelParam,
    _cnt_of_array( vxHeatmap_max_keypointKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxHeatmap_max_keypointInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_HEATMAP_MAX_KEYPOINT_list[] =
{
    &vxHeatmap_max_keypoint_CPU,
    &vxHeatmap_max_keypoint_VX,
    NULL
};
#ifdef __cplusplus
}
#endif

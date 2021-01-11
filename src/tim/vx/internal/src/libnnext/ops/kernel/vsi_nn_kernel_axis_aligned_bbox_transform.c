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
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_AXIS_ALIGNED_BBOX_TRANSFORM)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_AXIS_ALIGNED_BBOX_TRANSFORM)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_AXIS_ALIGNED_BBOX_TRANSFORM)
#define _VX_KERNEL_FUNC_KERNEL  (vxAxis_aligned_bbox_transformKernel)

typedef struct
{
    float x1, y1, x2, y2;
}BoxEncodingCorner;
typedef struct
{
    float w, h, x, y;
}BoxEncodingCenter;

void toBoxEncodingCorner
    (
    BoxEncodingCenter* ctr,
    BoxEncodingCorner* cnr
    )
{
    cnr->x1 = ctr->x - ctr->w / 2;
    cnr->y1 = ctr->y - ctr->h / 2;
    cnr->x2 = ctr->x + ctr->w / 2;
    cnr->y2 = ctr->y + ctr->h / 2;
}

void toBoxEncodingCenter
    (
    BoxEncodingCorner* cnr,
    BoxEncodingCenter* ctr
    )
{
    ctr->w = cnr->x2 - cnr->x1;
    ctr->h = cnr->y2 - cnr->y1;
    ctr->x = (cnr->x1 + cnr->x2) / 2;
    ctr->y = (cnr->y1 + cnr->y2) / 2;
}

static vsi_status VX_CALLBACK vxAxis_aligned_bbox_transformKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (0)
#define TENSOR_NUM_INPUT (4)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    int32_t* int32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t i;
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
        if (i == 2)
        {
            int32_in_buffer[i] = (int32_t *)vsi_nn_vxCopyTensorToData(context,
                input[i], &in_attr[i]);
        }
        else
        {
            f32_in_buffer[i] = (float *)malloc(in_elements[i] * sizeof(float));
            status = vsi_nn_vxConvertTensorToFloat32Data(
                context, input[i], &in_attr[i], f32_in_buffer[i],
                in_elements[i] * sizeof(float));
            TEST_CHECK_STATUS(status, final);
        }
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

    /* TODO: Add CPU kernel implement */
    {
        const uint32_t roiLength = 4;
        const uint32_t imageLength = 2;

        uint32_t numClasses = in_attr[1].size[0] / roiLength;
        uint32_t numRois = in_attr[0].size[1];
        uint32_t j;
        uint32_t roiIndex;
        for(roiIndex = 0; roiIndex < numRois; roiIndex++)
        {
            uint32_t batchIndex = int32_in_buffer[2][roiIndex];
            float imageHeight = f32_in_buffer[3][batchIndex * imageLength];
            float imageWidth = f32_in_buffer[3][batchIndex * imageLength + 1];
            BoxEncodingCorner roi_cnr;
            BoxEncodingCenter roiBefore;
            roi_cnr.x1 = f32_in_buffer[0][roiIndex * roiLength];
            roi_cnr.y1 = f32_in_buffer[0][roiIndex * roiLength + 1];
            roi_cnr.x2 = f32_in_buffer[0][roiIndex * roiLength + 2];
            roi_cnr.y2 = f32_in_buffer[0][roiIndex * roiLength + 3];
            toBoxEncodingCenter(&roi_cnr, &roiBefore);
            for (j = 0; j < numClasses; j++)
            {
                BoxEncodingCenter roi_ctr;
                BoxEncodingCorner roiAfter;
                BoxEncodingCorner cliped;
                uint32_t index = (roiIndex * numClasses + j) * roiLength;
                roi_ctr.w = (float)(exp(f32_in_buffer[1][index + 2]) * roiBefore.w);
                roi_ctr.h = (float)(exp(f32_in_buffer[1][index + 3]) * roiBefore.h);
                roi_ctr.x = roiBefore.x + f32_in_buffer[1][index] * roiBefore.w;
                roi_ctr.y = roiBefore.y + f32_in_buffer[1][index + 1] * roiBefore.h;
                toBoxEncodingCorner(&roi_ctr, &roiAfter);
                cliped.x1 = vsi_nn_min(vsi_nn_max(roiAfter.x1, 0.0f), imageWidth);
                cliped.y1 = vsi_nn_min(vsi_nn_max(roiAfter.y1, 0.0f), imageHeight);
                cliped.x2 = vsi_nn_min(vsi_nn_max(roiAfter.x2, 0.0f), imageWidth);
                cliped.y2 = vsi_nn_min(vsi_nn_max(roiAfter.y2, 0.0f), imageHeight);
                f32_out_buffer[0][index] = cliped.x1;
                f32_out_buffer[0][index + 1] = cliped.y1;
                f32_out_buffer[0][index + 2] = cliped.x2;
                f32_out_buffer[0][index + 3] = cliped.y2;
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
        if (int32_in_buffer[i]) free(int32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i]) free(f32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxAxis_aligned_bbox_transformKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxAxis_aligned_bbox_transformInitializer
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
vx_kernel_description_t vxAxis_aligned_bbox_transform_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxAxis_aligned_bbox_transformKernelParam,
    _cnt_of_array( vxAxis_aligned_bbox_transformKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxAxis_aligned_bbox_transform_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxAxis_aligned_bbox_transformKernelParam,
    _cnt_of_array( vxAxis_aligned_bbox_transformKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxAxis_aligned_bbox_transformInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_AXIS_ALIGNED_BBOX_TRANSFORM_list[] =
{
    &vxAxis_aligned_bbox_transform_CPU,
    &vxAxis_aligned_bbox_transform_VX,
    NULL
};
#ifdef __cplusplus
}
#endif

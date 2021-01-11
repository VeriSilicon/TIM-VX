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

#define _VX_KERNEL_VAR          (vx_kernel_GENERATE_PROPOSALS)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_GENERATE_PROPOSALS)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_GENERATE_PROPOSALS)
#define _VX_KERNEL_FUNC_KERNEL  (vxGenerate_proposalsKernel)

typedef struct
{
    float x1, y1, x2, y2;
}BoxEncodingCorner;
typedef struct
{
    float w, h, x, y;
}BoxEncodingCenter;

// toBoxEncodingCorner is implemented in vsi_nn_kernel_box_with_nms_limit.c
void toBoxEncodingCorner
    (
    BoxEncodingCenter* ctr,
    BoxEncodingCorner* cnr
    );

// toBoxEncodingCenter is implemented in vsi_nn_kernel_box_with_nms_limit.c
void toBoxEncodingCenter
    (
    BoxEncodingCorner* cnr,
    BoxEncodingCenter* ctr
    );

// iota is implemented in vsi_nn_kernel_detection_postprocess.c
static void _iota
    (
    int32_t * data,
    uint32_t len,
    int32_t value
    )
{
    uint32_t i;
    for (i = 0; i < len; i++)
    {
        data [i] = value;
        value++;
    }
}

// swap_element is implemented in vsi_nn_kernel_box_with_nms_limit.c
void swap_element
    (
    uint32_t* list,
    uint32_t first,
    uint32_t second
    );

// max_element is implemented in vsi_nn_kernel_box_with_nms_limit.c
uint32_t max_element
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    );

// getIoUAxisAligned is implemented in vsi_nn_kernel_box_with_nms_limit.c
float getIoUAxisAligned
    (
    const float* roi1,
    const float* roi2
    );

// sort_element_by_score is implemented in vsi_nn_kernel_box_with_nms_limit.c
void sort_element_by_score
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    );

void filterBoxes
    (
    const float* roiBase,
    const float* imageInfoBase,
    float minSize,
    uint32_t* select,
    uint32_t* len
    )
{
    const uint32_t kRoiDim = 4;
    uint32_t i = 0;
    uint32_t j;
    for(j = 0; j < *len; j++)
    {
        const float* roiInfo = roiBase + select[j] * kRoiDim;
        float roiWidth, roiHeight, xRoiCenter, yRoiCenter;
        roiWidth = roiInfo[2] - roiInfo[0];
        roiHeight = roiInfo[3] - roiInfo[1];
        xRoiCenter = roiInfo[0] + roiWidth / 2.0f;
        yRoiCenter = roiInfo[1] + roiHeight / 2.0f;
        if(roiWidth > minSize && roiHeight > minSize && xRoiCenter < imageInfoBase[1]
            && yRoiCenter < imageInfoBase[0])
        {
            select[i] = select[j];
            i++;
        }
    }
    *len = i;
}

static vsi_status VX_CALLBACK vxGenerate_proposalsKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (6)
#define TENSOR_NUM_INPUT (4)
#define TENSOR_NUM_OUTPUT (3)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    int32_t* int32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    float heightStride;
    float widthStride;
    int32_t preNmsTopN;
    int32_t postNmsTopN;
    float iouThreshold;
    float minSize;

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
        if(i < 2)
        {
            f32_out_buffer[i] = (float *)malloc(out_elements[i] * sizeof(float));
            memset(f32_out_buffer[i], 0, out_elements[i] * sizeof(float));
        }
        else
        {
            int32_out_buffer[i] = (int32_t *)malloc(out_elements[i] * sizeof(int32_t));
            memset(int32_out_buffer[i], 0, out_elements[i] * sizeof(int32_t));
        }
    }
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(heightStride),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(widthStride),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(preNmsTopN),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(postNmsTopN),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 4], &(iouThreshold),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 5], &(minSize),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    {
        uint32_t h, w, a, b, j;
        const uint32_t kRoiDim = 4;
        uint32_t numBatches = in_attr[0].size[3];
        uint32_t height = in_attr[0].size[2];
        uint32_t width = in_attr[0].size[1];
        uint32_t numAnchors = in_attr[0].size[0];
        uint32_t imageInfoLength = in_attr[3].size[0];

        uint32_t batchSize = height * width * numAnchors;
        uint32_t roiBufferSize = batchSize * kRoiDim;

        float * roiBuffer = (float*)malloc(roiBufferSize * sizeof(float));
        float * roiTransformedBuffer = (float*)malloc(roiBufferSize * sizeof(float));
        uint32_t* select = (uint32_t*)malloc(batchSize * sizeof(uint32_t));
        uint32_t index = 0;
        uint32_t scores_index = 0;
        uint32_t bboxDeltas_index = 0;
        uint32_t imageInfo_index = 0;
        uint32_t scores_out_index = 0;
        uint32_t roi_out_index = 0;

        // Compute the roi region for each anchor.
        for(h = 0; h < height; h++)
        {
            float hShift = h * heightStride;
            for(w = 0; w < width; w++)
            {
                float wShift = w * widthStride;
                uint32_t anchor_index = 0;
                for(a = 0; a < numAnchors; a++)
                {
                    roiBuffer[index] = f32_in_buffer[2][anchor_index] + wShift;
                    roiBuffer[index + 1] = f32_in_buffer[2][anchor_index + 1] + hShift;
                    roiBuffer[index + 2] = f32_in_buffer[2][anchor_index + 2] + wShift;
                    roiBuffer[index + 3] = f32_in_buffer[2][anchor_index + 3] + hShift;

                    index += kRoiDim;
                    anchor_index += kRoiDim;
                }
            }
        }

        for(b = 0; b < numBatches; b++)
        {
            const uint32_t roiLength = 4;

            uint32_t numRois = batchSize;
            uint32_t roiIndex;
            uint32_t select_len;
            int32_t numDetections = 0;
            for(roiIndex = 0; roiIndex < numRois; roiIndex++)
            {
                float imageHeight = f32_in_buffer[3][imageInfo_index];
                float imageWidth = f32_in_buffer[3][imageInfo_index + 1];
                BoxEncodingCorner roi_cnr;
                BoxEncodingCenter roiBefore;
                roi_cnr.x1 = roiBuffer[roiIndex * roiLength];
                roi_cnr.y1 = roiBuffer[roiIndex * roiLength + 1];
                roi_cnr.x2 = roiBuffer[roiIndex * roiLength + 2];
                roi_cnr.y2 = roiBuffer[roiIndex * roiLength + 3];
                toBoxEncodingCenter(&roi_cnr, &roiBefore);
                {
                    BoxEncodingCenter roi_ctr;
                    BoxEncodingCorner roiAfter;
                    BoxEncodingCorner cliped;
                    uint32_t idx = bboxDeltas_index + roiIndex * roiLength;
                    roi_ctr.w = (float)(exp(f32_in_buffer[1][idx + 2]) * roiBefore.w);
                    roi_ctr.h = (float)(exp(f32_in_buffer[1][idx + 3]) * roiBefore.h);
                    roi_ctr.x = roiBefore.x + f32_in_buffer[1][idx] * roiBefore.w;
                    roi_ctr.y = roiBefore.y + f32_in_buffer[1][idx + 1] * roiBefore.h;
                    toBoxEncodingCorner(&roi_ctr, &roiAfter);
                    cliped.x1 = vsi_nn_min(vsi_nn_max(roiAfter.x1, 0.0f), imageWidth);
                    cliped.y1 = vsi_nn_min(vsi_nn_max(roiAfter.y1, 0.0f), imageHeight);
                    cliped.x2 = vsi_nn_min(vsi_nn_max(roiAfter.x2, 0.0f), imageWidth);
                    cliped.y2 = vsi_nn_min(vsi_nn_max(roiAfter.y2, 0.0f), imageHeight);
                    roiTransformedBuffer[idx] = cliped.x1;
                    roiTransformedBuffer[idx + 1] = cliped.y1;
                    roiTransformedBuffer[idx + 2] = cliped.x2;
                    roiTransformedBuffer[idx + 3] = cliped.y2;
                }
            }

            // Find the top preNmsTopN scores.
            _iota((int32_t*)select, batchSize, 0);
            select_len = batchSize;
            if(preNmsTopN > 0 && preNmsTopN < (int32_t)batchSize)
            {
                sort_element_by_score(&(f32_in_buffer[0][scores_index]),
                    select, batchSize);
                select_len = preNmsTopN;
            }

            // Filter boxes, disgard regions with height or width < minSize.
            filterBoxes(roiTransformedBuffer, &(f32_in_buffer[3][0]),
                minSize, select, &select_len);

            // Apply hard NMS.
            if(postNmsTopN < 0)
            {
                postNmsTopN = select_len;
            }

            for(j = 0; (j < select_len && numDetections < postNmsTopN); j++)
            {
                // find max score and swap to the front.
                int32_t max_index = max_element(&(f32_in_buffer[0][scores_index]),
                    &(select[j]), select_len - j) + j;
                swap_element(select, max_index, j);

                // Calculate IoU of the rest, swap to the end (disgard) ifneeded.
                for(i = j + 1; i < select_len; i++)
                {
                    int32_t roiBase0 = select[i] * kRoiDim;
                    int32_t roiBase1 = select[j] * kRoiDim;
                    float iou = getIoUAxisAligned(&(roiTransformedBuffer[roiBase0]),
                        &(roiTransformedBuffer[roiBase1]));

                    if(iou >= iouThreshold)
                    {
                        swap_element(select, i, select_len - 1);
                        i--;
                        select_len--;
                    }
                }
                numDetections++;
            }

            for(i = 0; i < select_len; i++)
            {
                memcpy(&(f32_out_buffer[1][roi_out_index]),
                    &(roiTransformedBuffer[select[i] * kRoiDim]), kRoiDim * sizeof(float));
                f32_out_buffer[0][scores_out_index] =
                    f32_in_buffer[0][scores_index + select[i]];
                int32_out_buffer[2][scores_out_index] = b;
                scores_out_index++;
                roi_out_index += kRoiDim;
            }

            scores_index += batchSize;
            bboxDeltas_index += roiBufferSize;
            imageInfo_index += imageInfoLength;
        }

        vsi_nn_safe_free(roiBuffer);
        vsi_nn_safe_free(roiTransformedBuffer);
        vsi_nn_safe_free(select);
    }

    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if(i < 2)
        {
            status = vsi_nn_vxConvertFloat32DataToTensor(
                context, output[i], &out_attr[i], f32_out_buffer[i],
                out_elements[i] * sizeof(float));
            TEST_CHECK_STATUS(status, final);
        }
        else
        {
            vsi_nn_vxCopyDataToTensor(context, output[i], &out_attr[i],
                (uint8_t *)int32_out_buffer[i]);
        }
    }

final:
    for(i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if(f32_in_buffer[i]) free(f32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if(f32_out_buffer[i]) free(f32_out_buffer[i]);
        if(int32_out_buffer[i]) free(int32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxGenerate_proposalsKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxGenerate_proposalsInitializer
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
vx_kernel_description_t vxGenerate_proposals_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxGenerate_proposalsKernelParam,
    _cnt_of_array( vxGenerate_proposalsKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGenerate_proposals_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxGenerate_proposalsKernelParam,
    _cnt_of_array( vxGenerate_proposalsKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGenerate_proposalsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_GENERATE_PROPOSALS_list[] =
{
    &vxGenerate_proposals_CPU,
    &vxGenerate_proposals_VX,
    NULL
};
#ifdef __cplusplus
}
#endif

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
#include "utils/vsi_nn_link_list.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_BOX_WITH_NMS_LIMIT)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_BOX_WITH_NMS_LIMIT)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_BOX_WITH_NMS_LIMIT)
#define _VX_KERNEL_FUNC_KERNEL  (vxBox_with_nms_limitKernel)

static float hard_nms_kernel
    (
    float iou,
    float iouThreshold
    )
{
    return iou < iouThreshold ? 1.0f : 0.0f;
}

static float linear_nms_kernel
    (
    float iou,
    float iouThreshold
    )
{
    return iou < iouThreshold ? 1.0f : 1.0f - iou;
}

static float gaussian_nms_kernel
    (
    float iou,
    float sigma
    )
{
    return (float)(exp(-1.0f * iou * iou / sigma));
}

void swap_element
    (
    uint32_t* list,
    uint32_t first,
    uint32_t second
    )
{
    uint32_t temp = list[first];
    list[first] = list[second];
    list[second] = temp;
}

uint32_t max_element
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    )
{
    uint32_t i;
    uint32_t max_index = 0;
    float max_val = data[index_list[0]];
    for(i = 1; i < len; i++)
    {
        float val = data[index_list[i]];
        if (max_val < val)
        {
            max_val = val;
            max_index = i;
        }
    }
    return max_index;
}

static uint32_t max_comp_func
    (
    void* data,
    int32_t left,
    int32_t right
    )
{
    float* fdata = (float*)data;
    return fdata[left] >= fdata[right];
}

void sort_element_by_score
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    )
{
    vsi_nn_partition(data, 0, len - 1, max_comp_func, TRUE, index_list);
}

typedef struct
{
    float* fdata;
    uint32_t numClasses;
} class_comp_param;

static uint32_t class_comp_func
    (
    void* data,
    int32_t left,
    int32_t right
    )
{
    class_comp_param *p = (class_comp_param*)data;
    float* fdata = p->fdata;
    uint32_t numClasses = p->numClasses;
    uint32_t lhsClass = left % numClasses, rhsClass = right % numClasses;
    return lhsClass == rhsClass ? fdata[left] > fdata[right]
                : lhsClass < rhsClass;
}

static void sort_element_by_class
    (
    float* data,
    uint32_t* index_list,
    uint32_t len,
    uint32_t numClasses
    )
{
    class_comp_param class_comp;
    class_comp.fdata = data;
    class_comp.numClasses = numClasses;
    vsi_nn_partition(&class_comp, 0, len - 1, class_comp_func, TRUE, index_list);
}

// Taking two indices of bounding boxes, return the intersection-of-union.
float getIoUAxisAligned
    (
    const float* roi1,
    const float* roi2
    )
{
    const float area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1]);
    const float area2 = (roi2[2] - roi2[0]) * (roi2[3] - roi2[1]);
    const float x1 = vsi_nn_max(roi1[0], roi2[0]);
    const float x2 = vsi_nn_min(roi1[2], roi2[2]);
    const float y1 = vsi_nn_max(roi1[1], roi2[1]);
    const float y2 = vsi_nn_min(roi1[3], roi2[3]);
    const float w = vsi_nn_max(x2 - x1, 0.0f);
    const float h = vsi_nn_max(y2 - y1, 0.0f);
    const float areaIntersect = w * h;
    const float areaUnion = area1 + area2 - areaIntersect;
    return areaIntersect / areaUnion;
}

static vsi_status VX_CALLBACK vxBox_with_nms_limitKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (5)
#define TENSOR_NUM_INPUT (3)
#define TENSOR_NUM_OUTPUT (4)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    int32_t* int32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    int32_t* int32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    float scoreThreshold;
    int32_t maxNumDetections;
    int32_t nms_kernel_method;
    float iou_threshold;
    float sigma;
    float nms_score_threshold;

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
        if (i < 2)
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
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(scoreThreshold),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(maxNumDetections),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(nms_kernel_method),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(iou_threshold),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 4], &(sigma),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 5], &(nms_score_threshold),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    /* TODO: Add CPU kernel implement */
    {
        uint32_t j, n, b, c;
        const uint32_t kRoiDim = 4;
        uint32_t numRois = in_attr[0].size[1];
        uint32_t numClasses = in_attr[0].size[0];
        int32_t ind;

        uint32_t * batch_data = (uint32_t*)malloc(numRois * sizeof(uint32_t));
        int32_t numBatch = 0;
        uint32_t * select = NULL;
        uint32_t select_size = 0;
        uint32_t scores_index = 0;
        uint32_t roi_index = 0;
        uint32_t roi_out_index = 0;

        memset(batch_data, 0, numRois * sizeof(uint32_t));
        for (i = 0, ind = -1; i < numRois; i++)
        {
            if (int32_in_buffer[2][i] != ind)
            {
                ind = int32_in_buffer[2][i];
                numBatch++;
            }
            batch_data[numBatch - 1]++;
        }
        select = (uint32_t*)malloc(numBatch * numRois
            * numClasses * sizeof(uint32_t));
        memset(select, 0, numBatch * numRois * numClasses * sizeof(uint32_t));
        for (n = 0; n < (uint32_t)numBatch; n++)
        {
            int32_t numDetections_batch = 0;
            uint32_t select_start_batch = select_size;
            uint32_t select_len = 0;
            // Exclude class 0 (background)
            for (c = 1; c < numClasses; c++)
            {
                uint32_t select_start = select_size;
                int32_t maxNumDetections0 = maxNumDetections;
                uint32_t numDetections = 0;
                for (b = 0; b < batch_data[n]; b++)
                {
                    uint32_t index = b * numClasses + c;
                    float score = f32_in_buffer[0][scores_index + index];
                    if (score > scoreThreshold) {
                        select[select_size] = index;
                        select_size++;
                    }
                }
                select_len = select_size - select_start;

                if (maxNumDetections0 < 0)
                {
                    maxNumDetections0 = select_len;
                }

                for (j = 0; (j < select_len && numDetections < (uint32_t)maxNumDetections0); j++)
                {
                    // find max score and swap to the front.
                    int32_t max_index = max_element(&(f32_in_buffer[0][scores_index]),
                        &(select[select_start + j]), select_len - j) + j;

                    swap_element(&(select[select_start]), max_index, j);

                    // Calculate IoU of the rest, swap to the end (disgard) if needed.
                    for (i = j + 1; i < select_len; i++)
                    {
                        int32_t roiBase0 = roi_index + select[select_start + i] * kRoiDim;
                        int32_t roiBase1 = roi_index + select[select_start + j] * kRoiDim;
                        float iou = getIoUAxisAligned(&(f32_in_buffer[1][roiBase0]),
                            &(f32_in_buffer[1][roiBase1]));
                        float kernel_iou;
                        if (nms_kernel_method == 0)
                        {
                            kernel_iou = hard_nms_kernel(iou, iou_threshold);
                        }
                        else if (nms_kernel_method == 1)
                        {
                            kernel_iou = linear_nms_kernel(iou, iou_threshold);
                        }
                        else
                        {
                            kernel_iou = gaussian_nms_kernel(iou, sigma);

                        }
                        f32_in_buffer[0][scores_index + select[select_start + i]] *= kernel_iou;
                        if (f32_in_buffer[0][scores_index + select[select_start + i]] < nms_score_threshold)
                        {
                            swap_element(&(select[select_start]), i, select_len - 1);
                            i--;
                            select_len--;
                        }
                    }
                    numDetections++;
                }
                select_size = select_start + select_len;
                numDetections_batch += numDetections;
            }

            // Take top maxNumDetections.
            sort_element_by_score(&(f32_in_buffer[0][scores_index]), &(select[select_start_batch]),
                numDetections_batch);

            if (numDetections_batch > maxNumDetections)
            {
                select_size = select_start_batch + maxNumDetections;
            }
            select_len = select_size - select_start_batch;
            // Sort again by class.
            sort_element_by_class(&(f32_in_buffer[0][scores_index]), &(select[select_start_batch]),
                select_len, numClasses);

            for (i = 0; i < select_len; i++)
            {
                int32_t in_index0 = scores_index + select[select_start_batch + i];
                int32_t in_index1 = roi_index + select[select_start_batch + i] * kRoiDim;
                f32_out_buffer[0][roi_out_index] = f32_in_buffer[0][in_index0];
                memcpy(&(f32_out_buffer[1][roi_out_index * kRoiDim]),
                    &f32_in_buffer[1][in_index1], kRoiDim * sizeof(float));
                int32_out_buffer[2][roi_out_index] = select[select_start_batch + i] % numClasses;
                int32_out_buffer[3][roi_out_index] = n;
                roi_out_index++;
            }

            scores_index += batch_data[n] * numClasses;
            roi_index += batch_data[n] * numClasses * kRoiDim;
        }
        if (batch_data) free(batch_data);
        if (select) free(select);
    }

    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (i < 2)
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
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
        if (int32_in_buffer[i]) free(int32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i]) free(f32_out_buffer[i]);
        if (int32_out_buffer[i]) free(int32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxBox_with_nms_limitKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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

vx_status VX_CALLBACK vxBox_with_nms_limitInitializer
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
vx_kernel_description_t vxBox_with_nms_limit_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxBox_with_nms_limitKernelParam,
    _cnt_of_array( vxBox_with_nms_limitKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxBox_with_nms_limit_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxBox_with_nms_limitKernelParam,
    _cnt_of_array( vxBox_with_nms_limitKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxBox_with_nms_limitInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_BOX_WITH_NMS_LIMIT_list[] =
{
    &vxBox_with_nms_limit_CPU,
    &vxBox_with_nms_limit_VX,
    NULL
};
#ifdef __cplusplus
}
#endif

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
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (4)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.box_with_nms_limit")

/*
 * Kernel params
 */
static vx_param_description_t _box_with_nms_limit_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _BOX_WITH_NMS_LIMIT_PARAM_NUM  _cnt_of_array( _box_with_nms_limit_kernel_param_def )
#define SCORE_THRESHOLD         (7)
#define MAX_NUM_DETECTIONS      (8)
#define NMS_KERNEL_METHOD       (9)
#define IOU_THRESHOLD           (10)
#define SIGMA                   (11)
#define NMS_SCORE_THRESHOLD     (12)

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
    int32_t* int32_in_buffer[_INPUT_NUM] = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    int32_t* int32_out_buffer[_OUTPUT_NUM] = {0};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM];
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM];
    size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    size_t   out_elements[_OUTPUT_NUM] = {0};
    size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i = 0;
    float score_threshold = 0;
    int32_t max_num_detections = 0;
    int32_t nms_kernel_method = 0;
    float iou_threshold = 0;
    float sigma = 0;
    float nms_score_threshold = 0;
    uint32_t j = 0, n = 0, b = 0, c = 0;
    const uint32_t kRoiDim = 4;
    uint32_t numRois = 0;
    uint32_t numClasses = 0;
    int32_t ind = 0;
    uint32_t * batch_data = NULL;
    int32_t numBatch = 0;
    uint32_t * select = NULL;
    uint32_t select_size = 0;
    uint32_t scores_index = 0;
    uint32_t roi_index = 0;
    uint32_t roi_out_index = 0;

    /* prepare data */
    for (i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        if (i == 2)
        {
            int32_in_buffer[i] = (int32_t*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
            CHECK_PTR_FAIL_GOTO( int32_in_buffer[i], "Create input buffer fail.", final );
        }
        else
        {
            f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
            CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input buffer fail.", final );
        }
    }

    for (i = 0; i < _OUTPUT_NUM; i ++)
    {
        output[i] = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );
        vsi_nn_kernel_tensor_attr_get_stride( out_attr[i], out_stride_size[i] );
        out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
        out_bytes[i] = out_elements[i] * sizeof(float);
        if (i < 2)
        {
            f32_out_buffer[i] = (float *)malloc( out_bytes[i] );
            CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
            memset( f32_out_buffer[i], 0, out_bytes[i] );
        }
        else
        {
            int32_out_buffer[i] = (int32_t *)malloc( out_bytes[i] );
            CHECK_PTR_FAIL_GOTO( int32_out_buffer[i], "Create output buffer fail.", final );
            memset( int32_out_buffer[i], 0, out_bytes[i] );
        }
    }

#define VSI_NN_KERNEL_READ_SCALAR(type, idx, pointer) \
    vsi_nn_kernel_scalar_read_##type((vsi_nn_kernel_scalar_t)param[idx], pointer)

    status   = VSI_NN_KERNEL_READ_SCALAR(float32, SCORE_THRESHOLD, &score_threshold);
    status  |= VSI_NN_KERNEL_READ_SCALAR(int32, MAX_NUM_DETECTIONS, &max_num_detections);
    status  |= VSI_NN_KERNEL_READ_SCALAR(int32, NMS_KERNEL_METHOD, &nms_kernel_method);
    status  |= VSI_NN_KERNEL_READ_SCALAR(float32, IOU_THRESHOLD, &iou_threshold);
    status  |= VSI_NN_KERNEL_READ_SCALAR(float32, SIGMA, &sigma);
    status  |= VSI_NN_KERNEL_READ_SCALAR(float32, NMS_SCORE_THRESHOLD, &nms_score_threshold);
    CHECK_STATUS_FAIL_GOTO(status, final );
#undef VSI_NN_KERNEL_READ_SCALAR

    numRois = in_attr[0]->shape->data[1];
    numClasses = in_attr[0]->shape->data[0];

    batch_data = (uint32_t*)malloc(numRois * sizeof(uint32_t));
    CHECK_PTR_FAIL_GOTO( batch_data, "Create batch_data fail.", final );
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
    CHECK_PTR_FAIL_GOTO( select, "Create select fail.", final );
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
            int32_t maxNumDetections0 = max_num_detections;
            uint32_t numDetections = 0;
            for (b = 0; b < batch_data[n]; b++)
            {
                uint32_t index = b * numClasses + c;
                float score = f32_in_buffer[0][scores_index + index];
                if (score > score_threshold) {
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

        // Take top max_num_detections.
        sort_element_by_score(&(f32_in_buffer[0][scores_index]), &(select[select_start_batch]),
            numDetections_batch);

        if (numDetections_batch > max_num_detections && max_num_detections >= 0)
        {
            select_size = select_start_batch + max_num_detections;
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

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        if (i < 2)
        {
            status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                f32_out_buffer[i], out_elements[i] );
        }
        else
        {
            status = vsi_nn_kernel_tensor_write( output[i], out_attr[i],
                int32_out_buffer[i], out_bytes[i] );
        }
        CHECK_STATUS_FAIL_GOTO( status, final );
    }
final:
    vsi_nn_safe_free(batch_data);
    vsi_nn_safe_free(select);
    for (i = 0; i < _INPUT_NUM; i++)
    {
        vsi_nn_safe_free(f32_in_buffer[i]);
        vsi_nn_safe_free(int32_in_buffer[i]);

        if (in_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &in_attr[i] );
        }
    }
    for (i = 0; i < _OUTPUT_NUM; i++)
    {
        vsi_nn_safe_free(f32_out_buffer[i]);
        vsi_nn_safe_free(int32_out_buffer[i]);

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
    kernel->info.parameters  = _box_with_nms_limit_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _box_with_nms_limit_kernel_param_def );

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
    vsi_nn_kernel_node_param_t node_params[_BOX_WITH_NMS_LIMIT_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float score_threshold  = vsi_nn_kernel_param_get_float32( params, "score_threshold" );
    int32_t max_num_detections  = vsi_nn_kernel_param_get_int32( params, "max_num_detections" );
    int32_t nms_kernel_method  = vsi_nn_kernel_param_get_int32( params, "nms_kernel_method" );
    float iou_threshold  = vsi_nn_kernel_param_get_float32( params, "iou_threshold" );
    float sigma  = vsi_nn_kernel_param_get_float32( params, "sigma" );
    float nms_score_threshold  = vsi_nn_kernel_param_get_float32( params, "nms_score_threshold" );

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BOX_WITH_NMS_LIMIT_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCORE_THRESHOLD] = vsi_nn_kernel_scalar_create( graph, F32, &score_threshold );
            node_params[MAX_NUM_DETECTIONS] = vsi_nn_kernel_scalar_create( graph, I32, &max_num_detections );
            node_params[NMS_KERNEL_METHOD] = vsi_nn_kernel_scalar_create( graph, I32, &nms_kernel_method );
            node_params[IOU_THRESHOLD] = vsi_nn_kernel_scalar_create( graph, F32, &iou_threshold );
            node_params[SIGMA] = vsi_nn_kernel_scalar_create( graph, F32, &sigma );
            node_params[NMS_SCORE_THRESHOLD] = vsi_nn_kernel_scalar_create( graph, F32, &nms_score_threshold );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _BOX_WITH_NMS_LIMIT_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCORE_THRESHOLD] );
            vsi_nn_kernel_scalar_release( &node_params[MAX_NUM_DETECTIONS] );
            vsi_nn_kernel_scalar_release( &node_params[NMS_KERNEL_METHOD] );
            vsi_nn_kernel_scalar_release( &node_params[IOU_THRESHOLD] );
            vsi_nn_kernel_scalar_release( &node_params[SIGMA] );
            vsi_nn_kernel_scalar_release( &node_params[NMS_SCORE_THRESHOLD] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( box_with_nms_limit, _setup )

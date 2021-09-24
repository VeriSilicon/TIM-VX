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
#define _OUTPUT_NUM         (4)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.detect_post_nms")


/*
 * Kernel params
 */
static vx_param_description_t _detect_post_nms_kernel_param_def[] =
{
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
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _DETECT_POST_NMS_PARAM_NUM  _cnt_of_array( _detect_post_nms_kernel_param_def )

#define SCALAR_NMS_TYPE     (6)
#define SCALAR_MAX_NUM      (7)
#define SCALAR_MAX_CLASS    (8)
#define SCALAR_MAX_DETECT   (9)
#define SCALAR_SCORE_TH     (10)
#define SCALAR_IOU_TH       (11)
#define SCALAR_IS_BG        (12)

static void _swap_element
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

static uint32_t _max_element
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    )
{
    uint32_t i;
    uint32_t max_index = 0;
    float max_val = data[index_list[0]];
    for ( i = 1; i < len; i++ )
    {
        float val = data[index_list[i]];
        if ( max_val < val )
        {
            max_val = val;
            max_index = i;
        }
    }
    return max_index;
}

static float _getIoUAxisAligned
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

static uint32_t _max_comp_func
    (
    void* data,
    int32_t left,
    int32_t right
    )
{
    float* fdata = (float*)data;
    return fdata[left] >= fdata[right];
}

static void _sort_element_by_score
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    )
{
    vsi_nn_partition(data, 0, len - 1, _max_comp_func, TRUE, index_list);
}

static float _max_element_value
    (
    float* data,
    uint32_t len
    )
{
    uint32_t i;
    float max_val = data[0];
    for ( i = 1; i < len; i++ )
    {
        float val = data[i];
        if ( max_val < val )
        {
            max_val = val;
        }
    }
    return max_val;
}

static void _iota
    (
    int32_t * data,
    uint32_t len,
    int32_t value
    )
{
    uint32_t i;
    for ( i = 0; i < len; i++ )
    {
        data [i] = value;
        value++;
    }
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
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i, j;
    vsi_size_t  n, a, c, b, numBatches, numAnchors, numClasses;
    int32_t nms_type = 0;
    int32_t max_num_detections = 0;
    int32_t maximum_class_per_detection = 0;
    int32_t maximum_detection_per_class = 0;
    float   score_threshold  = 0.0f;
    float   iou_threshold    = 0.0f;
    int32_t is_bg_in_label   = 0;
    vsi_size_t numOutDetection = 0;

    /* prepare data */
    for ( i = 0; i < _INPUT_NUM; i++ )
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );

    }
    for ( i = 0; i < _OUTPUT_NUM; i++ )
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

    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_NMS_TYPE], &(nms_type));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_MAX_NUM], &(max_num_detections));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_MAX_CLASS], &(maximum_class_per_detection));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_MAX_DETECT], &(maximum_detection_per_class));
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_SCORE_TH], &(score_threshold));
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_IOU_TH], &(iou_threshold));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_IS_BG], &(is_bg_in_label));

    numBatches      = in_attr[0]->shape->data[2];
    numAnchors      = in_attr[0]->shape->data[1];
    numClasses      = in_attr[0]->shape->data[0];
    numOutDetection = out_attr[0]->shape->data[0];

    {
        vsi_size_t scores_index = 0;
        vsi_size_t scores_out_index = 0;
        uint32_t kRoiDim = 4;
        vsi_size_t roi_out_index = 0;
        vsi_size_t class_out_index = 0;
        uint32_t* select = (uint32_t*)malloc(numAnchors * numClasses * sizeof(uint32_t));
        float* maxScores = (float*)malloc(numAnchors * sizeof(float));
        uint32_t* scoreInds = (uint32_t*)malloc((numClasses - 1) * sizeof(uint32_t));

        for ( n = 0; n < numBatches; n++ )
        {
            float* roiBuffer = &(f32_in_buffer[1][n * numAnchors * kRoiDim]);
            if (nms_type)
            {
                uint32_t select_size = 0;
                uint32_t select_start = 0;
                uint32_t select_len = 0;
                uint32_t numDetections = 0;
                for ( c = 1; c < numClasses; c++ )
                {
                    select_start = select_size;
                    for ( b = 0; b < numAnchors; b++ )
                    {
                        const vsi_size_t index = b * numClasses + c;
                        float score = f32_in_buffer[0][scores_index + index];
                        if (score > score_threshold) {
                            select[select_size] = (uint32_t)index;
                            select_size++;
                        }
                    }
                    select_len = select_size - select_start;

                    if ( maximum_detection_per_class < 0 )
                    {
                        maximum_detection_per_class = select_len;
                    }
                    numDetections = 0;
                    for ( j = 0; (j < select_len && numDetections < (uint32_t)maximum_detection_per_class); j++ )
                    {
                        // find max score and swap to the front.
                        int32_t max_index = _max_element(&(f32_in_buffer[0][scores_index]),
                            &(select[select_start]), select_len);
                        _swap_element(&(select[select_start]), max_index, j);

                        // Calculate IoU of the rest, swap to the end (disgard) if needed.
                        for ( i = j + 1; i < select_len; i++ )
                        {
                            vsi_ssize_t roiBase0 = (select[select_start + i] / numClasses) * kRoiDim;
                            vsi_ssize_t roiBase1 = (select[select_start + j] / numClasses) * kRoiDim;
                            float iou = _getIoUAxisAligned(&(roiBuffer[roiBase0]),
                                &(roiBuffer[roiBase1]));

                            if ( iou >= iou_threshold )
                            {
                                _swap_element(&(select[select_start]), i, select_len - 1);
                                i--;
                                select_len--;
                            }
                        }
                        numDetections++;
                    }
                    select_size = select_start + numDetections;
                }

                select_len = select_size;
                select_start = 0;

                // Take top maxNumDetections.
                _sort_element_by_score(&(f32_in_buffer[0][scores_index]),
                    &(select[select_start]), select_len);

                for ( i = 0; i < select_len; i++ )
                {
                    uint32_t ind = select[i];
                    f32_out_buffer[0][scores_out_index + i] =
                        f32_in_buffer[0][scores_index + ind];
                    memcpy(&(f32_out_buffer[1][roi_out_index + i * kRoiDim]),
                        &roiBuffer[(ind / numClasses) * kRoiDim], kRoiDim * sizeof(float));
                    f32_out_buffer[2][class_out_index + i] = (float)((ind % numClasses)
                        - (is_bg_in_label ? 0 : 1));
                }
                f32_out_buffer[3][n] = (float)(select_len);
            }
            else
            {
                vsi_size_t numOutClasses = vsi_nn_min(numClasses - 1, (uint32_t)maximum_class_per_detection);
                uint32_t select_size = 0;
                uint32_t select_start = 0;
                uint32_t select_len = 0;
                uint32_t numDetections = 0;
                for ( a = 0; a < numAnchors; a++ )
                {
                    // exclude background class: 0
                    maxScores[a] = _max_element_value(&(f32_in_buffer[0]
                        [scores_index + a * numClasses + 1]), (uint32_t)(numClasses - 1));
                    if (maxScores[a] > score_threshold)
                    {
                            select[select_size] = (uint32_t)a;
                            select_size++;
                    }
                }
                select_len = select_size - select_start;

                if ( max_num_detections < 0 )
                {
                    max_num_detections = select_len;
                }
                for ( j = 0; (j < select_len && numDetections < (uint32_t)max_num_detections); j++ )
                {
                    // find max score and swap to the front.
                    int32_t max_index = _max_element(maxScores,
                        &(select[select_start + j]), select_len - j);
                    _swap_element(&(select[select_start]), max_index + j, j);

                    // Calculate IoU of the rest, swap to the end (disgard) if needed.
                    for ( i = j + 1; i < select_len; i++ )
                    {
                        int32_t roiBase0 = select[select_start + i] * kRoiDim;
                        int32_t roiBase1 = select[select_start + j] * kRoiDim;
                        float iou = _getIoUAxisAligned(&(roiBuffer[roiBase0]),
                            &(roiBuffer[roiBase1]));
                        if ( iou >= iou_threshold )
                        {
                            _swap_element(&(select[select_start]), i, select_len - 1);
                            i--;
                            select_len--;
                        }
                    }
                    numDetections++;
                }
                select_size = select_start + numDetections;
                select_len = select_size;

                for ( i = 0; i < select_len; i++ )
                {
                    _iota((int32_t*)scoreInds, (uint32_t)(numClasses - 1), 1);
                    _sort_element_by_score(&(f32_in_buffer[0][scores_index + select[i] * numClasses]),
                        scoreInds, (uint32_t)(numClasses - 1));
                    for (c = 0; c < numOutClasses; c++)
                    {
                        f32_out_buffer[0][scores_out_index + i * numOutClasses + c] =
                            f32_in_buffer[0][scores_index + select[i] * numClasses + scoreInds[c]];
                        memcpy(&(f32_out_buffer[1][roi_out_index + (i * numOutClasses + c)
                            * kRoiDim]), &roiBuffer[select[i] * kRoiDim], kRoiDim * sizeof(float));
                        f32_out_buffer[2][class_out_index + i * numOutClasses + c]
                            = (float)(scoreInds[c] - (is_bg_in_label ? 0 : 1));
                    }
                }
                f32_out_buffer[3][n] = (float)select_len;
            }
            scores_index += numAnchors * numClasses;
            scores_out_index += numOutDetection;
            roi_out_index += numOutDetection * kRoiDim;
            class_out_index += numOutDetection;
        }

        if (select) free(select);
        if (maxScores) free(maxScores);
        if (scoreInds) free(scoreInds);
    }
    /* save data */
    for ( i = 0; i < _OUTPUT_NUM; i++ )
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
    for ( i = 0; i < _OUTPUT_NUM; i++ )
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
    kernel->info.parameters  = _detect_post_nms_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _detect_post_nms_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_DETECT_POST_NMS_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t nms_type = vsi_nn_kernel_param_get_int32( params, "nms_type" );
    int32_t max_num_detections = vsi_nn_kernel_param_get_int32( params, "max_num_detections" );
    int32_t maximum_class_per_detection = vsi_nn_kernel_param_get_int32( params, "maximum_class_per_detection" );
    int32_t maximum_detection_per_class = vsi_nn_kernel_param_get_int32( params, "maximum_detection_per_class" );
    float   score_threshold  = vsi_nn_kernel_param_get_float32( params, "score_threshold" );
    float   iou_threshold    = vsi_nn_kernel_param_get_float32( params, "iou_threshold" );
    int32_t is_bg_in_label   = vsi_nn_kernel_param_get_int32( params, "is_bg_in_label" );

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _DETECT_POST_NMS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_NMS_TYPE]   = vsi_nn_kernel_scalar_create( graph, I32, &nms_type );
            node_params[SCALAR_MAX_NUM]    = vsi_nn_kernel_scalar_create( graph, I32, &max_num_detections );
            node_params[SCALAR_MAX_CLASS]  = vsi_nn_kernel_scalar_create( graph, I32, &maximum_class_per_detection );
            node_params[SCALAR_MAX_DETECT] = vsi_nn_kernel_scalar_create( graph, I32, &maximum_detection_per_class );
            node_params[SCALAR_SCORE_TH]   = vsi_nn_kernel_scalar_create( graph, F32, &score_threshold );
            node_params[SCALAR_IOU_TH]     = vsi_nn_kernel_scalar_create( graph, F32, &iou_threshold );
            node_params[SCALAR_IS_BG]      = vsi_nn_kernel_scalar_create( graph, I32, &is_bg_in_label );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _DETECT_POST_NMS_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_NMS_TYPE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MAX_NUM] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MAX_CLASS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MAX_DETECT] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCORE_TH] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_IOU_TH] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_IS_BG] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( detect_post_nms, _setup )


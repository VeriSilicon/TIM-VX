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
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (3)
 #define _CPU_IO_NUM        (_INPUT_NUM + _OUTPUT_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.nms")


/*
 * Kernel params
 */
static vx_param_description_t _nms_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define SCALAR_INPUT_MAX_SIZE          (5)
#define SCALAR_INPUT_IOU_THRES         (6)
#define SCALAR_INPUT_SCORE_THRES       (7)
#define SCALAR_INPUT_SOFT_NMS_SIGMA    (8)
#define _NMS_PARAM_NUM  _cnt_of_array( _nms_kernel_param_def )

typedef struct Candidate_s
{
    int index;
    float score;
    int suppress_begin_index;
}Candidate;
static void _swap_element
    (
    Candidate* list,
    uint32_t first,
    uint32_t second
    )
{
    Candidate temp;
    memcpy(&temp, &list[first], sizeof(Candidate));
    memcpy(&list[first], &list[second], sizeof(Candidate));
    memcpy(&list[second], &temp, sizeof(Candidate));
}

static uint32_t _max_element
    (
    Candidate* list,
    uint32_t len
    )
{
    uint32_t i;
    uint32_t max_index = 0;
    float max_val = list[0].score;
    for ( i = 1; i < len; i++ )
    {
        float val = list[i].score;
        if ( max_val < val )
        {
            max_val = val;
            max_index = i;
        }
    }

    return max_index;
}

typedef struct box_corner_encoding_s
{
  float y1;
  float x1;
  float y2;
  float x2;
}box_corner_encoding;

static float _computeIntersectionOverUnion
    (
    const float* boxes,
    const int32_t i,
    const int32_t j
    )
{
  box_corner_encoding box_i = ((box_corner_encoding *)boxes)[i];
  box_corner_encoding box_j = ((box_corner_encoding *)boxes)[j];
  const float box_i_y_min = vsi_nn_min(box_i.y1, box_i.y2);
  const float box_i_y_max = vsi_nn_max(box_i.y1, box_i.y2);
  const float box_i_x_min = vsi_nn_min(box_i.x1, box_i.x2);
  const float box_i_x_max = vsi_nn_max(box_i.x1, box_i.x2);
  const float box_j_y_min = vsi_nn_min(box_j.y1, box_j.y2);
  const float box_j_y_max = vsi_nn_max(box_j.y1, box_j.y2);
  const float box_j_x_min = vsi_nn_min(box_j.x1, box_j.x2);
  const float box_j_x_max = vsi_nn_max(box_j.x1, box_j.x2);

  const float area_i =
      (box_i_y_max - box_i_y_min) * (box_i_x_max - box_i_x_min);
  const float area_j =
      (box_j_y_max - box_j_y_min) * (box_j_x_max - box_j_x_min);
  const float intersection_ymax = vsi_nn_min(box_i_y_max, box_j_y_max);
  const float intersection_xmax = vsi_nn_min(box_i_x_max, box_j_x_max);
  const float intersection_ymin = vsi_nn_max(box_i_y_min, box_j_y_min);
  const float intersection_xmin = vsi_nn_max(box_i_x_min, box_j_x_min);
  const float intersection_area =
      vsi_nn_max(intersection_ymax - intersection_ymin, 0.0f) *
      vsi_nn_max(intersection_xmax - intersection_xmin, 0.0f);

  if (area_i <= 0 || area_j <= 0)
  {
      return 0.0f;
  }

  return intersection_area / (area_i + area_j - intersection_area);
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
    vsi_status status = VX_SUCCESS;
    vsi_nn_kernel_tensor_t tensors[_INPUT_NUM] = { NULL };
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float * buffer[_INPUT_NUM] = { NULL };
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_size_t stride_size[_INPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_size_t out_elements[_OUTPUT_NUM] = {0};
    vsi_nn_kernel_tensor_attr_t * attr[_INPUT_NUM] = { NULL };
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM] = {NULL};
    int32_t i = 0;
    int32_t num_boxes = 0;
    float* boxes = NULL;
    float* scores = NULL;
    float* selected_indices = NULL;
    float* selected_scores = NULL;
    float* num_selected_indices = NULL;
    Candidate * candidate = NULL;
    int32_t select_size = 0;
    int32_t max_output_size = 0;
    int32_t select_start = 0;
    int32_t select_len = 0;
    float iou_threshold = 0.f;
    float score_threshold = 0.f;
    float soft_nms_sigma = 0.f;
    float scale = 0;
    int32_t num_outputs = 0;

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_MAX_SIZE],
        &max_output_size);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_IOU_THRES],
        &iou_threshold);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_SCORE_THRES],
        &score_threshold);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_SOFT_NMS_SIGMA],
        &soft_nms_sigma);
    CHECK_STATUS_FAIL_GOTO(status, final );

    for ( i = 0;  i < _INPUT_NUM;  i++)
    {
        tensors[i]  = (vsi_nn_kernel_tensor_t)param[i];
        attr[i] = vsi_nn_kernel_tensor_attr_create( tensors[i] );

        vsi_nn_kernel_tensor_attr_get_stride( attr[i], stride_size[i] );
        buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[i], attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( buffer[i], "Create input buffer fail.", final );
    }

    for ( i = 0;  i < _OUTPUT_NUM;  i++)
    {
        output[i] = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );

        out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
        f32_out_buffer[i] = (float *)malloc( out_elements[i] * sizeof(float) );
        CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
        memset( f32_out_buffer[i], 0, out_elements[i] * sizeof(float) );
    }

    num_boxes = (int32_t)attr[0]->shape->data[1];
    boxes = buffer[0];
    scores = buffer[1];
    selected_indices = f32_out_buffer[0];
    selected_scores = f32_out_buffer[1];
    num_selected_indices = f32_out_buffer[2];

    candidate = (Candidate*)malloc(num_boxes * sizeof(Candidate));
    CHECK_PTR_FAIL_GOTO( candidate, "Create select buffer fail.", final );
    memset(candidate, 0, num_boxes * sizeof(Candidate));

    for (i = 0; i < num_boxes; ++i)
    {
        if (scores[i] > score_threshold)
        {
            candidate[select_size].index = i;
            candidate[select_size].score = scores[i];
            candidate[select_size].suppress_begin_index = 0;
            select_size++;
        }
    }

    num_outputs = vsi_nn_min(select_size, max_output_size);

    if (num_outputs == 0)
    {
        num_selected_indices[0] = 0;
    }

    if (soft_nms_sigma > 0.0f)
    {
        scale = -0.5f / soft_nms_sigma;
    }

    select_len = 0;
    while (select_len < num_outputs && select_start < select_size)
    {
        int32_t j = 0;
        float original_score = 0;
        vsi_bool should_hard_suppress = FALSE;

        // find max score and swap to the front.
        int32_t max_index = _max_element( &candidate[select_start], select_size - select_start);

        if (max_index != select_size - select_start - 1)
        {
            _swap_element(&(candidate[select_start]), max_index, 0);
        }

        original_score = candidate[select_start].score;
        // Calculate IoU of the rest, swap to the end (disgard) if needed.
        for ( j = select_len - 1; j >= candidate[select_start].suppress_begin_index; j-- )
        {
            int32_t idx = (int32_t)selected_indices[j];
            float iou = _computeIntersectionOverUnion(boxes, candidate[select_start].index, idx);

            // First decide whether to perform hard suppression.
            if (iou >= iou_threshold)
            {
                should_hard_suppress = TRUE;
                break;
            }

            // Suppress score if NMS sigma > 0.
            if (soft_nms_sigma > 0.0)
            {
                candidate[select_start].score =
                    candidate[select_start].score * (float)exp(scale * iou * iou);
            }

            if (candidate[select_start].score <= score_threshold)
                break;
        }

        candidate[select_start].suppress_begin_index = select_len;
        if (!should_hard_suppress)
        {
            if (candidate[select_start].score == original_score)
            {
                // Suppression has not occurred, so select next_candidate.
                selected_indices[select_len] = (float)candidate[select_start].index;
                selected_scores[select_len] = candidate[select_start].score;
                ++ select_len;
            }
            if ( candidate[select_start].score > score_threshold)
            {
                // Soft suppression might have occurred and current score is still
                // greater than score_threshold; add next_candidate back onto priority
                // queue.
                candidate[select_start].suppress_begin_index = select_len;
            }
        }

        select_start ++;
    }

    num_selected_indices[0] = (float)select_len;

    for ( i = select_len; i < max_output_size; i++)
    {
        selected_indices[i] = 0;
        selected_scores[i] = 0;
    }

    /* save data */
    for ( i = 0; i < _OUTPUT_NUM; i++ )
    {
        status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                f32_out_buffer[i], out_elements[i] );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    vsi_nn_safe_free(candidate);
    for( i = 0; i < _INPUT_NUM; i ++ )
    {
        if ( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
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
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _nms_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _nms_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_NMS_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t max_output_size = vsi_nn_kernel_param_get_int32(params, "max_output_size");
    float iou_threshold = vsi_nn_kernel_param_get_float32(params, "iou_threshold");
    float score_threshold = vsi_nn_kernel_param_get_float32(params, "score_threshold");
    float soft_nms_sigma = vsi_nn_kernel_param_get_float32(params, "soft_nms_sigma");

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _NMS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_INPUT_MAX_SIZE] = vsi_nn_kernel_scalar_create(
                    graph, I32, &max_output_size );
            node_params[SCALAR_INPUT_IOU_THRES] = vsi_nn_kernel_scalar_create(
                    graph, F32, &iou_threshold );
            node_params[SCALAR_INPUT_SCORE_THRES] = vsi_nn_kernel_scalar_create(
                    graph, F32, &score_threshold );
            node_params[SCALAR_INPUT_SOFT_NMS_SIGMA] = vsi_nn_kernel_scalar_create(
                    graph, F32, &soft_nms_sigma );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _NMS_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MAX_SIZE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_IOU_THRES] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCORE_THRES] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SOFT_NMS_SIGMA] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( nms, _setup )

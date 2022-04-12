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
#define _INPUT_NUM          (4)
#define _OUTPUT_NUM         (3)
 #define _TENSOR_NUM        (_INPUT_NUM + _OUTPUT_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.generate_proposals")


typedef struct vsi_nn_box_encoding_corner_t
{
    float x1, y1, x2, y2;
}vsi_nn_box_encoding_corner;

typedef struct vsi_nn_box_encoding_center_t
{
    float w, h, x, y;
}vsi_nn_box_encoding_center;
/*
 * Kernel params
 */
static vx_param_description_t _generate_proposals_kernel_param_def[] =
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
#define _GENERATE_PROPOSALS_PARAM_NUM  _cnt_of_array( _generate_proposals_kernel_param_def )


static void _to_box_encoding_corner
    (
    vsi_nn_box_encoding_center* ctr,
    vsi_nn_box_encoding_corner* cnr
    )
{
    cnr->x1 = ctr->x - ctr->w / 2;
    cnr->y1 = ctr->y - ctr->h / 2;
    cnr->x2 = ctr->x + ctr->w / 2;
    cnr->y2 = ctr->y + ctr->h / 2;
}

static void _to_box_encoding_center
    (
    vsi_nn_box_encoding_corner* cnr,
    vsi_nn_box_encoding_center* ctr
    )
{
    ctr->w = cnr->x2 - cnr->x1;
    ctr->h = cnr->y2 - cnr->y1;
    ctr->x = (cnr->x1 + cnr->x2) / 2;
    ctr->y = (cnr->y1 + cnr->y2) / 2;
}

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

void _filter_boxes
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
    uint32_t j = 0;

    for (j = 0; j < *len; j++)
    {
        const float* roiInfo = roiBase + select[j] * kRoiDim;
        float roiWidth, roiHeight, xRoiCenter, yRoiCenter;
        roiWidth = roiInfo[2] - roiInfo[0];
        roiHeight = roiInfo[3] - roiInfo[1];
        xRoiCenter = roiInfo[0] + roiWidth / 2.0f;
        yRoiCenter = roiInfo[1] + roiHeight / 2.0f;
        if (roiWidth > minSize && roiHeight > minSize && xRoiCenter < imageInfoBase[1]
            && yRoiCenter < imageInfoBase[0])
        {
            select[i] = select[j];
            i++;
        }
    }
    *len = i;
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
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i;
    float heightStride;
    float widthStride;
    int32_t preNmsTopN;
    int32_t postNmsTopN;
    float iouThreshold;
    float minSize;

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

    status = vsi_nn_kernel_scalar_read_float32( param[_TENSOR_NUM], &heightStride );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32( param[_TENSOR_NUM + 1], &widthStride );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[_TENSOR_NUM + 2], &preNmsTopN );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( param[_TENSOR_NUM + 3], &postNmsTopN );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32( param[_TENSOR_NUM + 4], &iouThreshold );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32( param[_TENSOR_NUM + 5], &minSize );
    CHECK_STATUS_FAIL_GOTO(status, final );

    {
        uint32_t h, w, a, b, j;
        const uint32_t kRoiDim = 4;
        vsi_size_t numBatches = in_attr[0]->shape->data[3];
        vsi_size_t height = in_attr[0]->shape->data[2];
        vsi_size_t width = in_attr[0]->shape->data[1];
        vsi_size_t numAnchors = in_attr[0]->shape->data[0];
        vsi_size_t imageInfoLength = in_attr[3]->shape->data[0];

        vsi_size_t batchSize = height * width * numAnchors;
        vsi_size_t roiBufferSize = batchSize * kRoiDim;

        float * roiBuffer = (float*)malloc(roiBufferSize * sizeof(float));
        float * roiTransformedBuffer = (float*)malloc(roiBufferSize * sizeof(float));
        uint32_t* select = (uint32_t*)malloc(batchSize * sizeof(uint32_t));
        uint32_t index = 0;
        vsi_size_t scores_index = 0;
        vsi_size_t bboxDeltas_index = 0;
        vsi_size_t imageInfo_index = 0;
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

        for (b = 0; b < numBatches; b++)
        {
            const uint32_t roiLength = 4;

            vsi_size_t numRois = batchSize;
            vsi_size_t roiIndex;
            uint32_t select_len;
            int32_t numDetections = 0;
            for (roiIndex = 0; roiIndex < numRois; roiIndex++)
            {
                float imageHeight = f32_in_buffer[3][imageInfo_index];
                float imageWidth = f32_in_buffer[3][imageInfo_index + 1];
                vsi_nn_box_encoding_corner roi_cnr;
                vsi_nn_box_encoding_center roiBefore;
                roi_cnr.x1 = roiBuffer[roiIndex * roiLength];
                roi_cnr.y1 = roiBuffer[roiIndex * roiLength + 1];
                roi_cnr.x2 = roiBuffer[roiIndex * roiLength + 2];
                roi_cnr.y2 = roiBuffer[roiIndex * roiLength + 3];
                _to_box_encoding_center(&roi_cnr, &roiBefore);
                {
                    vsi_nn_box_encoding_center roi_ctr;
                    vsi_nn_box_encoding_corner roiAfter;
                    vsi_nn_box_encoding_corner cliped;
                    vsi_size_t idx = bboxDeltas_index + roiIndex * roiLength;
                    roi_ctr.w = (float)(exp(f32_in_buffer[1][idx + 2]) * roiBefore.w);
                    roi_ctr.h = (float)(exp(f32_in_buffer[1][idx + 3]) * roiBefore.h);
                    roi_ctr.x = roiBefore.x + f32_in_buffer[1][idx] * roiBefore.w;
                    roi_ctr.y = roiBefore.y + f32_in_buffer[1][idx + 1] * roiBefore.h;
                    _to_box_encoding_corner(&roi_ctr, &roiAfter);
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
            _iota((int32_t*)select, (uint32_t)batchSize, 0);
            select_len = (uint32_t)batchSize;
            if(preNmsTopN > 0 && preNmsTopN < (int32_t)batchSize)
            {
                sort_element_by_score(&(f32_in_buffer[0][scores_index]),
                    select, (uint32_t)batchSize);
                select_len = preNmsTopN;
            }

            // Filter boxes, disgard regions with height or width < minSize.
            _filter_boxes(roiTransformedBuffer, &(f32_in_buffer[3][0]),
                minSize, select, &select_len);

            // Apply hard NMS.
            if (postNmsTopN < 0)
            {
                postNmsTopN = select_len;
            }

            for (j = 0; (j < select_len && numDetections < postNmsTopN); j++)
            {
                // find max score and swap to the front.
                int32_t max_index = max_element(&(f32_in_buffer[0][scores_index]),
                    &(select[j]), select_len - j) + j;
                swap_element(select, max_index, j);

                // Calculate IoU of the rest, swap to the end (disgard) ifneeded.
                for (i = j + 1; i < select_len; i++)
                {
                    int32_t roiBase0 = select[i] * kRoiDim;
                    int32_t roiBase1 = select[j] * kRoiDim;
                    float iou = getIoUAxisAligned(&(roiTransformedBuffer[roiBase0]),
                        &(roiTransformedBuffer[roiBase1]));

                    if (iou >= iouThreshold)
                    {
                        swap_element(select, i, select_len - 1);
                        i--;
                        select_len--;
                    }
                }
                numDetections++;
            }

            for (i = 0; i < select_len; i++)
            {
                memcpy(&(f32_out_buffer[1][roi_out_index]),
                    &(roiTransformedBuffer[select[i] * kRoiDim]), kRoiDim * sizeof(float));
                f32_out_buffer[0][scores_out_index] =
                    f32_in_buffer[0][scores_index + select[i]];
                f32_out_buffer[2][scores_out_index] = (float)b;
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
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _generate_proposals_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _generate_proposals_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_GENERATE_PROPOSALS_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    float height_stride = vsi_nn_kernel_param_get_float32( params, "height_stride");
    float width_stride = vsi_nn_kernel_param_get_float32( params, "width_stride");
    int32_t pre_nms_top_n = vsi_nn_kernel_param_get_int32( params, "pre_nms_top_n");
    int32_t post_nms_top_n = vsi_nn_kernel_param_get_int32( params, "post_nms_top_n");
    float iou_threshold = vsi_nn_kernel_param_get_float32(params, "iou_threshold");
    float min_size = vsi_nn_kernel_param_get_float32(params, "min_size");

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GENERATE_PROPOSALS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[_TENSOR_NUM    ] = vsi_nn_kernel_scalar_create( graph, F32, &height_stride );
            node_params[_TENSOR_NUM + 1] = vsi_nn_kernel_scalar_create( graph, F32, &width_stride );
            node_params[_TENSOR_NUM + 2] = vsi_nn_kernel_scalar_create( graph, I32, &pre_nms_top_n );
            node_params[_TENSOR_NUM + 3] = vsi_nn_kernel_scalar_create( graph, I32, &post_nms_top_n );
            node_params[_TENSOR_NUM + 4] = vsi_nn_kernel_scalar_create( graph, F32, &iou_threshold );
            node_params[_TENSOR_NUM + 5] = vsi_nn_kernel_scalar_create( graph, F32, &min_size );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GENERATE_PROPOSALS_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &node_params[_TENSOR_NUM    ] );
            vsi_nn_kernel_scalar_release( &node_params[_TENSOR_NUM + 1] );
            vsi_nn_kernel_scalar_release( &node_params[_TENSOR_NUM + 2] );
            vsi_nn_kernel_scalar_release( &node_params[_TENSOR_NUM + 3] );
            vsi_nn_kernel_scalar_release( &node_params[_TENSOR_NUM + 4] );
            vsi_nn_kernel_scalar_release( &node_params[_TENSOR_NUM + 5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( generate_proposals, _setup )

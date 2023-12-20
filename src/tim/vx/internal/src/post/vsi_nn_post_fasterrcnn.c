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
#include <math.h>

#include "vsi_nn_context.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_node_attr_template.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"
#include "post/vsi_nn_post_fasterrcnn.h"
#include "vsi_nn_error.h"

/*
    faster-rcnn default image classes -- 21
*/
static const char* FASTER_RCNN_CLASSES[] =
    {"__background__",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"};
#define VSI_NN_FASTERRCNN_CLASSES_NUM _cnt_of_array(FASTER_RCNN_CLASSES)

static void _dump_boxes
    (
    vsi_nn_fasterrcnn_box_t **box,
    vsi_nn_fasterrcnn_param_t *param
    );

static vsi_status _fill_fasterrcnn_param
    (
    vsi_nn_graph_t *graph,
    vsi_nn_fasterrcnn_param_t *param
    );

static vsi_status _fill_fasterrcnn_inputs
    (
    vsi_nn_graph_t *graph,
    vsi_nn_fasterrcnn_param_t *param,
    vsi_nn_fasterrcnn_inputs_t *inputs
    );

static vsi_status _unscale_roi
    (
    float *rois,
    vsi_nn_fasterrcnn_param_t *param
    );

static vsi_status _bbox_transform_inv
    (
    float *rois,
    float *bbox,
    vsi_nn_fasterrcnn_param_t *param,
    float **boxes
    );

static float detection_box_iou
    (
    float *A,
    float *B
    );

static void detection_box_nms
    (
    float *box,
    float thresh,
    uint32_t rois_num,
    uint32_t *keep,
    uint32_t *num
    );

static void detection_box_qsort
    (
    float *box,
    int32_t start,
    int32_t end
    );

static void _init_box
    (
    vsi_nn_link_list_t *node
    );

static void _dump_boxes
    (
    vsi_nn_fasterrcnn_box_t **box,
    vsi_nn_fasterrcnn_param_t *param
    );

static vsi_status _fill_fasterrcnn_param
    (
    vsi_nn_graph_t *graph,
    vsi_nn_fasterrcnn_param_t *param
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t i;
    vsi_nn_node_t   *node;
    vsi_nn_tensor_t *tensor;

    if(NULL == graph || NULL == param)
    {
        return status;
    }

    tensor = NULL;

    for(i=0; i<graph->node_num; i++)
    {
        node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );
        //printf("i[%u] op[%s]\n", i, vsi_nn_OpGetName(node->op));
        if (node && node->op == VSI_NN_OP_PROPOSAL)
        {
            memcpy(&param->iminfo, &node->nn_param.proposal.im_info,
                    sizeof(vsi_nn_proposal_im_info));
            tensor = vsi_nn_GetTensor(graph,node->output.tensors[0]);
            CHECK_PTR_FAIL_GOTO( tensor, "Get tensor fail.", final );

            param->rois_num = (uint32_t)tensor->attr.size[1];
        }
    }

    if(0 == param->rois_num)
    {
        VSILOGE("Can not find [Proposal] layer in network");
        return status;
    }
    status = VSI_SUCCESS;

    /* fill default parameters */
#define VSI_NN_FASTERRCNN_DEF_CONF_THRESH (0.7f)
#define VSI_NN_FASTERRCNN_DEF_NMS_THRESH (0.3f)
    param->conf_thresh = VSI_NN_FASTERRCNN_DEF_CONF_THRESH;
    param->nms_thresh = VSI_NN_FASTERRCNN_DEF_NMS_THRESH;
    param->classes_num = VSI_NN_FASTERRCNN_CLASSES_NUM;
    param->classes = FASTER_RCNN_CLASSES;

final:
    return status;
} /* _fill_fasterrcnn_param() */

static vsi_status _fill_fasterrcnn_inputs
    (
    vsi_nn_graph_t *graph,
    vsi_nn_fasterrcnn_param_t *param,
    vsi_nn_fasterrcnn_inputs_t *inputs
    )
{
    vsi_status status;
    uint32_t i,rois_num,size[2],dim;
    vsi_nn_tensor_t *tensor;

    if(NULL == graph || NULL == inputs)
    {
        return VSI_FAILURE;
    }

    status = VSI_FAILURE;
    tensor = NULL;
    rois_num = param->rois_num;
    for(i=0; i<graph->output.num; i++)
    {
        /* bbox [84,rois] */
        /* cls  [21,rois] */
        /* rois [5,rois] */
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
        CHECK_PTR_FAIL_GOTO( tensor, "get tensor fail.", final );
        size[0] = (uint32_t)tensor->attr.size[0];
        size[1] = (uint32_t)tensor->attr.size[1];
        dim = tensor->attr.dim_num;
        if(dim == 2 && size[1] == rois_num)
        {
            switch (size[0])
            {
            case 5:
                inputs->rois = tensor;
                break;
            case 21:
                inputs->cls = tensor;
                break;
            case 84:
                inputs->bbox = tensor;
                break;
            default:
                break;
            }
        }
    }

final:
    if(inputs->rois == NULL ||
       inputs->cls == NULL ||
       inputs->bbox == NULL)
    {
        VSILOGE("Can not find [rois,cls,bbox] tensor in network");
        return status;
    }
    status = VSI_SUCCESS;

    return status;
} /* _fill_fasterrcnn_inputs() */

static vsi_status _unscale_roi
    (
    float *rois,
    vsi_nn_fasterrcnn_param_t *param
    )
{
    uint32_t i;
    float *data;

    data = rois;
    for(i=0; i<param->rois_num; i++)
    {
        data[1] = data[1] / param->iminfo.scale[0];
        data[2] = data[2] / param->iminfo.scale[1];
        data[3] = data[3] / param->iminfo.scale[0];
        data[4] = data[4] / param->iminfo.scale[1];

        data += 5;
    }

    return VSI_SUCCESS;
}

static vsi_status _bbox_transform_inv
    (
    float *rois,
    float *bbox,
    vsi_nn_fasterrcnn_param_t *param,
    float **boxes
    )
{
    float *pred_boxes = NULL, *ppred = NULL;
    float *proi,*pbbox;
    uint32_t i,j,rois_num,bbox_num,class_num;
    float img_w,img_h;
    vsi_status status = VSI_FAILURE;

    float w,h,ctr_x,ctr_y;
    float dx,dy,dw,dh;
    float pred_ctr_x,pred_ctr_y,pred_w,pred_h;

    img_w = param->iminfo.size[0];
    img_h = param->iminfo.size[1];
    rois_num = param->rois_num;
    class_num = param->classes_num;
    bbox_num = class_num * 4;
    pred_boxes = (float *)malloc(sizeof(float) * rois_num * bbox_num);
    CHECK_PTR_FAIL_GOTO( pred_boxes, "Create buffer fail.", final );
    status = VSI_SUCCESS;

    proi = rois;
    pbbox = bbox;
    ppred = pred_boxes;
    for(i=0; i<rois_num; i++)
    {
        /* roi_data {0,x1,y1,x2,y2} */
        w = proi[3] - proi[1] + 1.0f;
        h = proi[4] - proi[2] + 1.0f;
        ctr_x = proi[1] + 0.5f * w;
        ctr_y = proi[2] + 0.5f * h;

        /* bbox {rois_num,84} */
        for(j=0; j<class_num; j++)
        {
            dx = pbbox[0];
            dy = pbbox[1];
            dw = pbbox[2];
            dh = pbbox[3];

            pred_ctr_x = dx * w + ctr_x;
            pred_ctr_y = dy * h + ctr_y;
            pred_w = expf(dw) * w;
            pred_h = expf(dh) * h;

            /* update upper-left corner location */
            ppred[0] = pred_ctr_x - 0.5f * pred_w;
            ppred[1] = pred_ctr_y - 0.5f * pred_h;

            /* update lower-right corner location */
            ppred[2] = pred_ctr_x + 0.5f * pred_w;
            ppred[3] = pred_ctr_y + 0.5f * pred_h;

            /* adjust new corner locations to be within the image region */
            ppred[0] = vsi_nn_max(0.0f, vsi_nn_min(ppred[0], img_w - 1.0f));
            ppred[1] = vsi_nn_max(0.0f, vsi_nn_min(ppred[1], img_h - 1.0f));
            ppred[2] = vsi_nn_max(0.0f, vsi_nn_min(ppred[2], img_w - 1.0f));
            ppred[3] = vsi_nn_max(0.0f, vsi_nn_min(ppred[3], img_h - 1.0f));

            pbbox += 4;
            ppred += 4;
        }

        proi += 5;
    }

    *boxes = pred_boxes;

final:
    return status;
}

static float detection_box_iou
    (
    float *A,
    float *B
    )
{
    float x1,y1,x2,y2,width,height,area,A_area,B_area;

    if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1])
    {
        return 0;
    }

    /* overlapped region (=box) */
    x1 = vsi_nn_max(A[0], B[0]);
    y1 = vsi_nn_max(A[1], B[1]);
    x2 = vsi_nn_min(A[2], B[2]);
    y2 = vsi_nn_min(A[3], B[3]);

    /* intersection area */
    width    = vsi_nn_max(0.0f, x2 - x1 + 1.0f);
    height   = vsi_nn_max(0.0f, y2 - y1 + 1.0f);
    area     = width * height;

    /* area of A, B */
    A_area   = (A[2] - A[0] + 1.0f) * (A[3] - A[1] + 1.0f);
    B_area   = (B[2] - B[0] + 1.0f) * (B[3] - B[1] + 1.0f);

    /* IOU */
    return area / (A_area + B_area - area);
}

static void detection_box_nms
    (
    float *box,
    float thresh,
    uint32_t rois_num,
    uint32_t *keep,
    uint32_t *num
    )
{
    uint32_t i,j;
    uint32_t *is_dead = NULL;

    is_dead = (uint32_t *)malloc(sizeof(uint32_t) * rois_num);
    CHECK_PTR_FAIL_GOTO( is_dead, "Create buffer fail.", final );
    memset(is_dead, 0, sizeof(uint32_t) * rois_num);

    for(i = 0; i < rois_num; i++)
    {
        if(is_dead[i])
        {
            continue;
        }

        for(j = i + 1; j < rois_num; ++j)
        {
            if(!is_dead[j] && detection_box_iou(&box[i * 5], &box[j * 5]) > thresh)
            {
                is_dead[j] = 1;
            }
        }
    }

    j = 0;
    for(i=0; i<rois_num; i++)
    {
        if(!is_dead[i])
        {
            keep[j] = i;
            j++;
        }
    }
    *num = j;

final:
    vsi_nn_safe_free(is_dead);
}

static void detection_box_qsort
    (
    float *box,
    int32_t start,
    int32_t end
    )
{
    /*
        box[x] = {x1, y1, x2, y2, score};
    */
    int i;
    float pivot_score = box[start * 5 + 4];
    int32_t left = start + 1, right = end;
    float temp[5];

    while (left <= right)
    {
        while(left <= end && box[left * 5 + 4] >= pivot_score)
            ++left;
        while (right > start && box[right * 5 + 4] <= pivot_score)
            --right;

        if (left <= right)
        {
            /* swap box */
            for(i = 0; i < 5; ++i)
            {
                temp[i] = box[left * 5 + i];
            }
            for(i = 0; i < 5; ++i)
            {
                box[left * 5 + i] = box[right * 5 + i];
            }
            for(i = 0; i < 5; ++i)
            {
                box[right * 5 + i] = temp[i];
            }

            ++left;
            --right;
        }
    }

    if (right > start)
    {
        for(i = 0; i < 5; ++i)
        {
            temp[i] = box[start * 5 + i];
        }
        for(i = 0; i < 5; ++i)
        {
            box[start * 5 + i] = box[right * 5 + i];
        }
        for(i = 0; i < 5; ++i)
        {
            box[right * 5 + i] = temp[i];
        }
    }

    if(start < right - 1)
    {
        detection_box_qsort(box, start, right - 1);
    }
    if(right + 1 < end)
    {
        detection_box_qsort(box, right + 1, end);
    }
}

static void _init_box(vsi_nn_link_list_t *node)
{
    vsi_nn_fasterrcnn_box_t *box = NULL;
    box = (vsi_nn_fasterrcnn_box_t *)node;
    memset(box, 0, sizeof(vsi_nn_fasterrcnn_box_t));
}

static vsi_status _fasterrcnn_post_process
    (
    float *rois,
    float *bbox,
    float *cls,
    vsi_nn_fasterrcnn_param_t *param,
    vsi_nn_fasterrcnn_box_t **dets_box
    )
{
    vsi_status status;
    uint32_t i,j,k;
    uint32_t rois_num,classes_num;
    float *pred_boxes = NULL,*dets = NULL;
    float *pdets = NULL, *ppred = NULL;
    vsi_nn_fasterrcnn_box_t *box = NULL;
    uint32_t *keep = NULL,num;
    float score;

    if(NULL == rois || NULL == bbox || NULL == cls || NULL == param)
    {
        return VSI_FAILURE;
    }

    status = VSI_FAILURE;
    status = _unscale_roi(rois, param);
    if(status != VSI_SUCCESS)
    {
        VSILOGE("unscale roi fail");
        return status;
    }

    status = _bbox_transform_inv(rois, bbox, param, &pred_boxes);
    if(status != VSI_SUCCESS)
    {
        VSILOGE("transform bbox fail");
        return status;
    }

    rois_num = param->rois_num;
    classes_num = param->classes_num;
    dets = (float *)malloc(sizeof(float) * 5 * rois_num);
    if(NULL == dets)
    {
        status = VSI_FAILURE;
        goto final;
    }

    keep = NULL;
    keep = (uint32_t *)malloc(sizeof(uint32_t) * rois_num);
    if(NULL == keep)
    {
        status = VSI_FAILURE;
        goto final;
    }

    /* i=1, skip background */
    for(i=1; i<param->classes_num; i++)
    {
        /* pred_boxes{rois_num,84} */
        pdets = dets;
        ppred = pred_boxes + 4 * i;
        for(j=0; j<rois_num; j++)
        {
            pdets[0] = ppred[0];
            pdets[1] = ppred[1];
            pdets[2] = ppred[2];
            pdets[3] = ppred[3];
            pdets[4] = cls[j*classes_num + i];

            pdets += 5;
            ppred += classes_num*4;
        }

        detection_box_qsort(dets, 0, rois_num - 1);

        num = 0;
        memset(keep, 0, sizeof(int32_t) * rois_num);
        detection_box_nms(dets, param->nms_thresh, rois_num, keep, &num);

        for(k=0; k<num; k++)
        {
            score = dets[keep[k]*5+4];
            if(score > param->conf_thresh)
            {
                if(NULL != dets_box)
                {
                    box = (vsi_nn_fasterrcnn_box_t *)
                        vsi_nn_LinkListNewNode(sizeof(vsi_nn_fasterrcnn_box_t), _init_box);
                    CHECK_PTR_FAIL_GOTO( box, "Create box fail.", final );
                    box->score = dets[keep[k]*5+4];
                    box->class_id = i;
                    box->x1 = dets[keep[k]*5+0];
                    box->y1 = dets[keep[k]*5+1];
                    box->x2 = dets[keep[k]*5+2];
                    box->y2 = dets[keep[k]*5+3];
                    vsi_nn_LinkListPushStart(
                        (vsi_nn_link_list_t **)dets_box,
                        (vsi_nn_link_list_t *)box );
                }
            }
        }
    }

final:
    if(keep)free(keep);
    if(dets)free(dets);
    if(pred_boxes)free(pred_boxes);
    return status;
} /* _fasterrcnn_post_process() */

static void _dump_boxes
    (
    vsi_nn_fasterrcnn_box_t **box,
    vsi_nn_fasterrcnn_param_t *param
    )
{
    vsi_nn_fasterrcnn_box_t *iter = *box;

    while (iter)
    {
        if(param->classes)
        {
            VSILOGI(" classes[%s] score[%f] coordinate[%f %f %f %f]",
                param->classes[iter->class_id],
                iter->score,
                iter->x1, iter->y1, iter->x2, iter->y2);
        }
        else
        {
            VSILOGI(" classes_id[%u] score[%f] coordinate[%f %f %f %f]",
                iter->class_id,
                iter->score,
                iter->x1, iter->y1, iter->x2, iter->y2);
        }


        iter = (vsi_nn_fasterrcnn_box_t *)
            vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter );
    }
}

vsi_status vsi_nn_FasterRCNN_PostProcess
    (
    vsi_nn_graph_t *graph,
    vsi_nn_fasterrcnn_inputs_t *inputs,
    vsi_nn_fasterrcnn_param_t *param,
    vsi_nn_fasterrcnn_box_t **dets_box
    )
{
    vsi_status status;
    vsi_nn_fasterrcnn_inputs_t frcnn_inputs;
    vsi_nn_fasterrcnn_param_t frcnn_param;
    float *roi_data,*bbox_data,*cls_data;

    if(NULL == graph)
    {
        return VSI_FAILURE;
    }

    status = VSI_FAILURE;
    memset(&frcnn_inputs, 0, sizeof(vsi_nn_fasterrcnn_inputs_t));
    memset(&frcnn_param, 0, sizeof(vsi_nn_fasterrcnn_param_t));

    if(NULL == param)
    {
        status = _fill_fasterrcnn_param(graph, &frcnn_param);
        if(status != VSI_SUCCESS)
        {
            VSILOGE("Auto fill faster-rcnn parameters fail");
            return status;
        }
    }
    else
    {
        memcpy(&frcnn_param, param, sizeof(vsi_nn_fasterrcnn_param_t));
    }

    if(NULL == inputs)
    {
        status = _fill_fasterrcnn_inputs(graph, &frcnn_param, &frcnn_inputs);
        if(status != VSI_SUCCESS)
        {
            VSILOGE("Auto fill faster-rcnn inputs fail");
            return status;
        }
    }
    else
    {
        memcpy(&frcnn_inputs, inputs, sizeof(vsi_nn_fasterrcnn_inputs_t));
    }

    roi_data = NULL,bbox_data = NULL, cls_data = NULL;
    roi_data  = vsi_nn_ConvertTensorToFloat32Data(graph, frcnn_inputs.rois);
    bbox_data = vsi_nn_ConvertTensorToFloat32Data(graph, frcnn_inputs.bbox);
    cls_data  = vsi_nn_ConvertTensorToFloat32Data(graph, frcnn_inputs.cls);

    status = _fasterrcnn_post_process(
        roi_data,
        bbox_data,
        cls_data,
        &frcnn_param,
        dets_box
        );
    if(status != VSI_SUCCESS)
    {
        goto final;
    }

    _dump_boxes(dets_box, &frcnn_param);

    status = VSI_SUCCESS;
final:
    if(roi_data)free(roi_data);
    if(bbox_data)free(bbox_data);
    if(cls_data)free(cls_data);
    return status;
}

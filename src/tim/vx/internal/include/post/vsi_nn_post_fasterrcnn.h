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
#ifndef _VSI_NN_POST_FASTERRCNN_H_
#define _VSI_NN_POST_FASTERRCNN_H_

#include "vsi_nn_types.h"
#include "vsi_nn_node_type.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_link_list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_fasterrcnn_box_t
{
    vsi_nn_link_list_t link_list;

    /* upper-left coordinate(x1,y1) */
    float x1;
    float y1;
    /* lower-right coordinate(x2,y2) */
    float x2;
    float y2;
    float score;
    uint32_t class_id;
} VSI_PUBLIC_TYPE vsi_nn_fasterrcnn_box_t;

typedef struct _vsi_nn_fasterrcnn_param_t
{
    float conf_thresh;
    float nms_thresh;
    const char **classes;
    uint32_t classes_num;
    uint32_t rois_num;
    vsi_nn_proposal_im_info iminfo;
} VSI_PUBLIC_TYPE vsi_nn_fasterrcnn_param_t;

typedef struct _vsi_nn_fasterrcnn_inputs_t
{
    vsi_nn_tensor_t *rois;
    vsi_nn_tensor_t *cls;
    vsi_nn_tensor_t *bbox;
} VSI_PUBLIC_TYPE vsi_nn_fasterrcnn_inputs_t;

OVXLIB_API vsi_status vsi_nn_FasterRCNN_PostProcess
    (
    vsi_nn_graph_t *graph,
    vsi_nn_fasterrcnn_inputs_t *inputs,
    vsi_nn_fasterrcnn_param_t *param,
    vsi_nn_fasterrcnn_box_t **dets_box
    );

#ifdef __cplusplus
}
#endif

#endif

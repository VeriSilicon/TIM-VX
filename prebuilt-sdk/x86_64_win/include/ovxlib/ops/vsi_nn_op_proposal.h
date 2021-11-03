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
#ifndef _VSI_NN_OP_PROPOSAL_H
#define _VSI_NN_OP_PROPOSAL_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_proposal_lcl_data
{
    vx_tensor rois;
    vx_tensor score;
} vsi_nn_proposal_lcl_data;

typedef struct _vsi_nn_proposal_anchor
{
    float * ratio;
    float * scale;
    int32_t ratio_num;
    int32_t scale_num;
    int32_t base_size;
} vsi_nn_proposal_anchor;

typedef struct _vsi_nn_proposal_im_info
{
    float  size[2];
    float  scale[2];
} vsi_nn_proposal_im_info;

typedef struct _vsi_nn_proposal_param
{
    vsi_nn_proposal_lcl_data local;
    vsi_nn_proposal_anchor  anchor;
    vsi_nn_proposal_im_info im_info;
    uint32_t   feat_stride;
    uint32_t   pre_nms_topn;
    uint32_t   post_nms_topn;
    float      nms_thresh;
    uint32_t   min_size;
} vsi_nn_proposal_param;

#ifdef __cplusplus
}
#endif

#endif


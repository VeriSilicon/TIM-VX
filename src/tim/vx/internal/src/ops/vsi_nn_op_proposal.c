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
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"

/*
*  inputs[0] - scores
*  inputs[1] - bboxs
*  inputs[2] - im_info
*  inputs[3] - anchors
*  outputs[0] - rois
*  outputs[1] - scores
*/

#define ROUND(x)        ((int)(x + 0.5f))

static vsi_nn_tensor_t * create_im_info_tensor
    (
    vsi_nn_graph_t * graph,
    vsi_nn_proposal_im_info * im_info
    )
{
    vsi_nn_tensor_t * tensor;
    vsi_nn_tensor_attr_t attr;
    memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    attr.size[0] = 1;
    attr.size[1] = 1;
    attr.size[2] = 4;
    attr.size[3] = 1;
    attr.dim_num = 4;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    tensor = vsi_nn_CreateTensorFromData( graph,
        (uint8_t *)im_info, &attr );
    if( NULL == tensor )
    {
        VSILOGE( "Create im info tensor fail." );
    }
    return tensor;
} /* create_im_info_tensor() */


static vsi_nn_tensor_t * create_anchor_tensor
    (
    vsi_nn_graph_t * graph,
    vsi_nn_proposal_anchor * anchor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * tensor;
    float      * p_anchor;
    float      * data_anchor;
    float        base_area;
    float        center;
    uint32_t         anchor_sz;
    float        ratio_h;
    float        ratio_w;
    float        scale_h;
    float        scale_w;
    int               i;
    int               j;

    if( NULL == anchor->ratio || NULL == anchor->scale
        || 0 >= anchor->ratio_num || 0 >= anchor->scale_num
        || 0 >= anchor->base_size )
    {
        VSILOGE( "Create anchor tensor fail." );
        return NULL;
    }

    memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    anchor_sz = anchor->ratio_num * anchor->scale_num * 4;

    data_anchor = (float *)malloc( anchor_sz * sizeof( float ) );
    if( NULL == data_anchor )
    {
        VSILOGE( "Create anchor tensor fail." );
        return NULL;
    }

    /* Generate anchor data */
    p_anchor = data_anchor;
    base_area = (float)(anchor->base_size * anchor->base_size);
    center = (float)(0.5f * (anchor->base_size - 1.0f));

    for( i = 0; i < anchor->ratio_num; i ++ )
    {
        ratio_w = (float)ROUND( sqrt( base_area / anchor->ratio[i] ) );
        ratio_h = (float)ROUND( ratio_w * anchor->ratio[i] );
        for( j = 0; j < anchor->scale_num; j ++ )
        {
            scale_w = (float)( 0.5f * (ratio_w * anchor->scale[j] - 1.0f ) );
            scale_h = (float)( 0.5f * (ratio_h * anchor->scale[j] - 1.0f ) );
            p_anchor[0] = center - scale_w;
            p_anchor[1] = center - scale_h;
            p_anchor[2] = center + scale_w;
            p_anchor[3] = center + scale_h;
            p_anchor += 4;
        }
    }

    /* Create tensor */
    attr.size[0] = 1;
    attr.size[1] = 1;
    attr.size[2] = 4;
    attr.size[3] = anchor->ratio_num * anchor->scale_num;
    attr.dim_num = 4;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    tensor = vsi_nn_CreateTensorFromData( graph,
        (uint8_t *)data_anchor, &attr );

    free( data_anchor );

    if( NULL == tensor )
    {
        VSILOGE( "Create anchor tensor fail." );
    }
    return tensor;
} /* create_im_info_tensor() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_rpn_params_t p;
    vx_tensor rois_tmp, score_tmp;

    status = VSI_FAILURE;
    if(self->nn_param.proposal.local.rois)
    {
        rois_tmp = self->nn_param.proposal.local.rois;
    }
    else
    {
        rois_tmp = outputs[0]->t;
    }
    if(self->nn_param.proposal.local.score)
    {
        score_tmp = self->nn_param.proposal.local.score;
    }
    else
    {
        score_tmp = (NULL != outputs[1])?outputs[1]->t : NULL;
    }

    p.feature_stride = self->nn_param.proposal.feat_stride;
    p.min_size = self->nn_param.proposal.min_size;
    p.pre_nms_topn = self->nn_param.proposal.pre_nms_topn;
    p.post_nms_topn = self->nn_param.proposal.post_nms_topn;
    p.nms_thresh = self->nn_param.proposal.nms_thresh;

    self->n = vxRPNLayer(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->t,
        inputs[3]->t,
        inputs[2]->t,
        &p,
        sizeof( vx_nn_rpn_params_t ),
        rois_tmp,
        score_tmp
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensor_t * im_info;
    vsi_nn_tensor_t * anchors;

    im_info = inputs[2];
    anchors = inputs[3];
    /* Check and generate im_info */
    if( NULL == im_info )
    {
        im_info = create_im_info_tensor( node->graph,
            &node->nn_param.proposal.im_info );
        inputs[2] = im_info;
        node->input.tensors[2] = vsi_nn_AttachTensorToGraph(
            node->graph, VSI_NN_TENSOR_ID_AUTO, im_info );
    }

    /* Check and generate anchors */
    if( NULL == anchors )
    {
        anchors = create_anchor_tensor( node->graph,
            &node->nn_param.proposal.anchor );
        inputs[3] = anchors;
        node->input.tensors[3] = vsi_nn_AttachTensorToGraph(
            node->graph, VSI_NN_TENSOR_ID_AUTO, anchors );
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = 5;
        outputs[0]->attr.size[1] = node->nn_param.proposal.post_nms_topn;
        outputs[0]->attr.dim_num = 2;
    }

    if( NULL != outputs[1] && VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num )
    {
        outputs[1]->attr.size[0] = 1;
        outputs[1]->attr.size[1] = node->nn_param.proposal.post_nms_topn;
        outputs[1]->attr.dim_num = 2;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_size_t size[VSI_NN_MAX_DIM_NUM];
    uint32_t dim;
    vx_tensor rois_tmp, score_tmp;

    rois_tmp = NULL, score_tmp = NULL;
    if( direction == VSI_NN_OPTIMIZE_BACKWARD )
    {
        VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);

        dim = 4;
        size[0] = 1;
        size[1] = 1;
        self->nn_param.proposal.local.rois = NULL;
        self->nn_param.proposal.local.score = NULL;
        /* reshape rois tensor, [5,roi_num] --> [1,1,5,roi_num] */
        if(2 == outputs[0]->attr.dim_num)
        {
            size[2] = outputs[0]->attr.size[0];
            size[3] = outputs[0]->attr.size[1];
#ifdef VSI_40BIT_VA_SUPPORT
            rois_tmp = vxReshapeTensor(outputs[0]->t, size, dim);
#else
            {
                vsi_size_t i;
                int32_t size_32bit[VSI_NN_MAX_DIM_NUM];
                for(i = 0; i< VSI_NN_MAX_DIM_NUM; i++)
                {
                    size_32bit[i] = (int32_t)size[i];
                }
                rois_tmp = vxReshapeTensor(outputs[0]->t, size_32bit, dim);
            }
#endif
            if(NULL == rois_tmp)
            {
                goto error;
            }
            self->nn_param.proposal.local.rois = rois_tmp;
        }

        /* reshape score tensor, [1,roi_num] --> [1,1,1,roi_num] */
        if(outputs[1] != NULL && 2 == outputs[1]->attr.dim_num)
        {
            size[2] = outputs[1]->attr.size[0];
            size[3] = outputs[1]->attr.size[1];
#ifdef VSI_40BIT_VA_SUPPORT
            score_tmp = vxReshapeTensor(outputs[1]->t, size, dim);
#else
            {
                vsi_size_t i;
                int32_t size_32bit[VSI_NN_MAX_DIM_NUM];
                for(i = 0; i< VSI_NN_MAX_DIM_NUM; i++)
                {
                    size_32bit[i] = (int32_t)size[i];
                }
                score_tmp = vxReshapeTensor(outputs[1]->t, size_32bit, dim);
            }
#endif
            if(NULL == score_tmp)
            {
                goto error;
            }
            self->nn_param.proposal.local.score = score_tmp;
        }
    }

    return VSI_SUCCESS;
error:
    if(rois_tmp)vxReleaseTensor(&rois_tmp);
    if(score_tmp)vxReleaseTensor(&score_tmp);
    return VSI_FAILURE;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vx_tensor rois = self->nn_param.proposal.local.rois;
    vx_tensor score = self->nn_param.proposal.local.score;
    if( NULL != self && NULL != self->n )
    {
        if(rois)
        {
            vxReleaseTensor(&rois);
            rois = NULL;
        }
        if(score)
        {
            vxReleaseTensor(&score);
            score = NULL;
        }
        vxReleaseNode( &self->n );
        self->n = NULL;
    }
    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PROPOSAL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 4,
    /* output_num */ 2
    );
#ifdef __cplusplus
}
#endif

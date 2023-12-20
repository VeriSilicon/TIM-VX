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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_roi_pool_params_ext_t params;
    vx_tensor rois_input;

    status = VSI_FAILURE;
    params.khr.pool_type = self->nn_param.roi_pool.type;
    params.spatial_scale = self->nn_param.roi_pool.scale;
    params.pooled_width = self->nn_param.roi_pool.size[0];
    params.pooled_height = self->nn_param.roi_pool.size[1];

    if(self->nn_param.roi_pool.local.rois)
    {
        rois_input = self->nn_param.roi_pool.local.rois;
    }
    else
    {
        rois_input = inputs[1]->t;
    }

    self->n = vxROIPoolingLayer(
        self->graph->g,
        inputs[0]->t,
        rois_input,
        (vx_nn_roi_pool_params_t *)&params,
        sizeof( vx_nn_roi_pool_params_ext_t ),
        outputs[0]->t
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
    BEGIN_IO_TYPE_DECL(ROI_POOL, 2, 1)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16, D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16, D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32, D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_F16, D_F16)
        IO_TYPE(D_F16,  D_F16, D_F32)
        IO_TYPE(D_F16,  D_F32, D_F16)
        IO_TYPE(D_F16,  D_F32, D_F32)
        IO_TYPE(D_F32,  D_F16, D_F16)
        IO_TYPE(D_F32,  D_F16, D_F32)
        IO_TYPE(D_F32,  D_F32, D_F16)
        IO_TYPE(D_F32,  D_F32, D_F32)
        IO_TYPE(D_BF16, D_BF16, D_F32)
        IO_TYPE(D_BF16, D_BF16, D_BF16)
        IO_TYPE(D_F32,  D_F32,  D_BF16)

        /* HW 9.0 */
        IO_TYPE(D_BF16,  D_F16,  D_BF16)
    END_IO_TYPE_DECL(ROI_POOL)
    if(!VALIDATE_OP_IO_TYPES(ROI_POOL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_add_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = 4;
        outputs[0]->attr.size[0] = node->nn_param.roi_pool.size[0];
        outputs[0]->attr.size[1] = node->nn_param.roi_pool.size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];

        //FIXME: old proposal outputs dimension is 4
        if(4 == inputs[1]->attr.dim_num)
        {
            outputs[0]->attr.size[3] = inputs[1]->attr.size[3];
        }
        else
        {
            outputs[0]->attr.size[3] = inputs[1]->attr.size[1];
        }
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
    vx_tensor rois_tmp;

    VSI_UNREFERENCED(outputs);

    rois_tmp = NULL;
    if( direction == VSI_NN_OPTIMIZE_FORWARD && inputs[1]->attr.dim_num == 2 )
    {
        /* reshape proposal rois tensor, [5,roi_num] --> [1,1,5,roi_num] */
        VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);

        dim = 4;
        size[0] = 1;
        size[1] = 1;
        self->nn_param.roi_pool.local.rois = NULL;
        /* reshape rois tensor, [5,roi_num] --> [1,1,5,roi_num] */
        if(2 == inputs[1]->attr.dim_num)
        {
            size[2] = inputs[1]->attr.size[0];
            size[3] = inputs[1]->attr.size[1];
            rois_tmp = vsi_nn_safe_reshape_tensor(inputs[1]->t, (void*)size, (vsi_size_t)dim, sizeof(size[0]));
            if(NULL == rois_tmp)
            {
                return VSI_FAILURE;
            }
            self->nn_param.proposal.local.rois = rois_tmp;
        }
    }

    return VSI_SUCCESS;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vx_tensor rois = NULL;
    if ( NULL != self && NULL != self->n )
    {
        rois = self->nn_param.roi_pool.local.rois;
        if(rois)
        {
            vxReleaseTensor(&rois);
            rois = NULL;
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
    /* op_name    */ ROI_POOL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

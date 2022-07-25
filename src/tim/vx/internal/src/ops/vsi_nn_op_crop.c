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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpCheck(VSI_NN_OP_STRIDED_SLICE, self, inputs, outputs);

    return ret;
}

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_crop_param * p = NULL;
    int32_t i = 0;
    uint32_t j = 0;
    vsi_nn_internal_node_t* curr = NULL;

    vsi_nn_internal_init_node_wksp( self );
    p = (vsi_nn_crop_param *)&(self->nn_param.crop);

    if (p->axis >= (int32_t)inputs[0]->attr.dim_num)
    {
        VSILOGE("Invalid parameter: axis!\n");
        return FALSE;
    }

    if ( VSI_NN_DIM_AUTO != outputs[0]->attr.dim_num )
    {
        goto final;
    }

    if (p->dims + p->axis == inputs[0]->attr.dim_num)
    {
        for (i = 0; i < p->axis; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
        for (i = p->axis; i < (int32_t)inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    else
    {
        if (p->dims == 1)
        {
            for (i = 0; i <= p->axis; i++)
            {
                outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
                p->offset[i] = p->offset[0];
            }
            for (i = p->axis + 1; i < (int32_t)inputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
            }
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        }
        else
        {
            VSILOGE("Invalid parameter: offset dims!\n");
            return FALSE;
        }
    }

final:
    for (j = 0; j < self->nn_param.crop.dims; j++)
    {
        p->lcl_data->begin_dims[j] = (int32_t)self->nn_param.crop.offset[j];
        p->lcl_data->end_dims[j] = (int32_t)self->nn_param.crop.offset[j] + (int32_t)outputs[0]->attr.size[j];
        p->lcl_data->stride_dims[j] = 1;
    }

    for (j = self->nn_param.crop.dims; j < inputs[0]->attr.dim_num; j++)
    {
        p->lcl_data->begin_dims[j] = 0;
        p->lcl_data->end_dims[j] = (int32_t)outputs[0]->attr.size[j];
        p->lcl_data->stride_dims[j] = 1;
    }

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_STRIDED_SLICE, 0, 0 );
    curr->node->nn_param.strided_slice.begin_dims = p->lcl_data->begin_dims;
    curr->node->nn_param.strided_slice.begin_dims_num = inputs[0]->attr.dim_num;
    curr->node->nn_param.strided_slice.end_dims = p->lcl_data->end_dims;
    curr->node->nn_param.strided_slice.end_dims_num = inputs[0]->attr.dim_num;
    curr->node->nn_param.strided_slice.stride_dims = p->lcl_data->stride_dims;
    curr->node->nn_param.strided_slice.stride_dims_num = inputs[0]->attr.dim_num;
    curr->node->nn_param.strided_slice.begin_mask = 0;
    curr->node->nn_param.strided_slice.end_mask = 0;
    curr->node->nn_param.strided_slice.shrink_axis_mask = 0;
    curr->node->nn_param.strided_slice.new_axis_mask = 0;
    curr->inputs[0] = inputs[0];
    curr->outputs[0] = outputs[0];
    vsi_nn_internal_setup_node( self, curr );

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_crop_param * p = NULL;

    p = &(self->nn_param.crop);

    p->lcl_data = (vsi_nn_crop_lcl_data *)malloc(sizeof(vsi_nn_crop_lcl_data));
    if (NULL == p->lcl_data)
    {
        return  VSI_FAILURE;
    }
    memset(p->lcl_data, 0, sizeof(vsi_nn_crop_lcl_data));

    return status;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_crop_param * p = NULL;

    p = &(self->nn_param.crop);

    vsi_nn_safe_free(p->lcl_data);

    vsi_nn_internal_deinit_node_wksp( self );
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */



#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CROP,
    /* init       */ op_init,
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

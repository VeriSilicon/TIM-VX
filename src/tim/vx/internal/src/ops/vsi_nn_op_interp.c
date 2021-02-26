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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_internal_compute_node( self );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_interp_param *p = NULL;

    p = &self->nn_param.interp;

    if ((p->pad_beg > 0) || (p->pad_end > 0))
    {
        VSILOGE("Only supports non-pos padding (cropping) for now ");
        return FALSE;
    }

    return TRUE;
} /* op_check() */


static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_interp_param *p = NULL;
    int32_t height_in_eff_, width_in_eff_;
    int32_t height_out, width_out;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t *crop_tensor = NULL;
    vsi_nn_tensor_t *crop_in_tensor = NULL;
    float factor = 1.0f;
    int32_t pad_beg = 0;
    int32_t pad_end = 0;

    if ( NULL == self )
    {
        return FALSE;
    }

    p = &self->nn_param.interp;
    pad_beg = -p->pad_beg;
    pad_end = -p->pad_end;
    width_in_eff_  = inputs[0]->attr.size[0] + p->pad_beg + p->pad_end;
    height_in_eff_ = inputs[0]->attr.size[1] + p->pad_beg + p->pad_end;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        memcpy( outputs[0]->attr.size, inputs[0]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
        if ((p->shrink_factor > 0) && (p->zoom_factor <= 0))
        {
            width_out  = (width_in_eff_ - 1) / p->shrink_factor + 1;
            height_out = (height_in_eff_ - 1) / p->shrink_factor + 1;
        }
        else if ((p->zoom_factor > 0) && (p->shrink_factor <= 0))
        {
            width_out  = (width_in_eff_ - 1) * (p->zoom_factor - 1) + width_in_eff_;
            height_out = (height_in_eff_ - 1) * (p->zoom_factor - 1) + height_in_eff_;
        }
        else if ((p->height > 0) && (p->width > 0))
        {
            width_out  = p->width;
            height_out = p->height;
        }
        else if ((p->zoom_factor > 0) && (p->shrink_factor > 0))
        {
            width_out  = (width_in_eff_ - 1) / p->shrink_factor + 1;
            height_out = (height_in_eff_ - 1) / p->shrink_factor + 1;
            width_out  = (width_out - 1) * (p->zoom_factor - 1) + width_out;
            height_out = (height_out - 1) * (p->zoom_factor - 1) + height_out;
        }
        else if (NULL != inputs[1])
        {
            width_out  = inputs[1]->attr.size[0];
            height_out = inputs[1]->attr.size[1];
        }
        else
        {
            VSILOGE("Not support params ");
            return FALSE;
        }

        if ((width_out < 0) || (height_out < 0) || (width_in_eff_ < 0) || (height_in_eff_ < 0))
        {
            VSILOGE("value shoud be positive: width_out %d height_out %d width_in_eff_ %d height_in_eff_ %d ",
                    width_out, height_out, width_in_eff_, height_in_eff_);
            return FALSE;
        }

        outputs[0]->attr.size[0] = width_out;
        outputs[0]->attr.size[1] = height_out;
    }

    factor = (float)(outputs[0]->attr.size[0]) / (float)(width_in_eff_);

    if ((pad_beg > 0) || (pad_end > 0))
    {
        vsi_nn_tensor_attr_t attr;
        int32_t use_virtual_tensor = 1;
        int32_t *begin_dims;
        int32_t *end_dims;
        int32_t *stride_dims;
        uint32_t i;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        vsi_nn_internal_init_tensor_attr(&attr, &inputs[0]->attr.dtype, use_virtual_tensor);
        crop_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        crop_in_tensor = crop_tensor->t;
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_STRIDED_SLICE, 1, 1 );
        curr->node->nn_param.strided_slice.begin_dims_num = inputs[0]->attr.dim_num;
        curr->node->nn_param.strided_slice.end_dims_num = inputs[0]->attr.dim_num;
        curr->node->nn_param.strided_slice.stride_dims_num = inputs[0]->attr.dim_num;
        curr->node->nn_param.strided_slice.begin_mask = 0;
        curr->node->nn_param.strided_slice.end_mask = 0;
        curr->node->nn_param.strided_slice.shrink_axis_mask = 0;
        begin_dims = (int32_t *)vsi_nn_internal_new_node_param(curr,
            VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        end_dims   = (int32_t *)vsi_nn_internal_new_node_param(curr,
            VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        stride_dims  = (int32_t *)vsi_nn_internal_new_node_param(curr,
            VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            stride_dims[i] = 1;
        }

        begin_dims[0] = pad_beg;
        begin_dims[1] = pad_beg;
        end_dims[0]   = inputs[0]->attr.size[0] - pad_end;
        end_dims[1]   = inputs[0]->attr.size[1] - pad_end;

        if (inputs[0]->attr.dim_num > 2)
        {
            for (i = 2 ; i < inputs[0]->attr.dim_num; i++)
            {
                begin_dims[i] = 0;
                end_dims[i]   = inputs[0]->attr.size[i];
            }
        }
        curr->node->nn_param.strided_slice.begin_dims = begin_dims;
        curr->node->nn_param.strided_slice.end_dims = end_dims;
        curr->node->nn_param.strided_slice.stride_dims = stride_dims;
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = crop_in_tensor;
        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        crop_in_tensor = inputs[0];
    }

    if ((width_in_eff_ == (int32_t)outputs[0]->attr.size[0]) && (height_in_eff_ == (int32_t)outputs[0]->attr.size[1]))
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 1, 1 );
        curr->inputs[0]  = crop_in_tensor;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESIZE_INTERNAL, 1, 1 );
        curr->node->nn_param.resize_internal.align_corners = vx_true_e;
        curr->node->nn_param.resize_internal.factor = factor;
        curr->node->nn_param.resize_internal.half_pixel_centers = vx_false_e;
        curr->inputs[0]  = crop_in_tensor;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
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
    vsi_status     status;

    status = VSI_SUCCESS;
    vsi_nn_internal_optimize_node( self, direction );

    return status;
} /* op_optimize() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_internal_init_node_wksp(self);
    self->nn_param.interp.height         = 0;
    self->nn_param.interp.width          = 0;
    self->nn_param.interp.pad_beg        = 0;
    self->nn_param.interp.pad_end        = 0;
    self->nn_param.interp.shrink_factor  = 0;
    self->nn_param.interp.zoom_factor    = 0;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp(self);
    status = vsi_nn_op_common_deinit(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ INTERP,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS


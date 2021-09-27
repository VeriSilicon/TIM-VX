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
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static vsi_bool _is_same_shape
    (
    vsi_nn_tensor_t * inputs,
    vsi_size_t *sizes,
    uint32_t dims
    )
{
    uint32_t i = 0;

    if (inputs->attr.dim_num != dims)
        return FALSE;

    for (i = 0; i < dims; i++)
    {
        if (sizes[i] != inputs->attr.size[i])
            return FALSE;
    }

    return TRUE;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;

    status = vsi_nn_internal_compute_node( self );

    return status;
} /* op_compute() */


static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if ( _is_same_shape(inputs[0], outputs[0]->attr.size, outputs[0]->attr.dim_num) )
    {
        return vsi_nn_internal_optimize_node(self, direction );
    }
    else
    {
        return VSI_SUCCESS;
    }
} /* op_optimize() */

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
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    float factor = self->nn_param.resize_1d.factor;
    vsi_nn_internal_node_t* curr = NULL;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        if (factor != 0)
        {
            outputs[0]->attr.size[0] = (uint32_t)(inputs[0]->attr.size[0] * factor);
        }
        else
        {
            outputs[0]->attr.size[0] = self->nn_param.resize_1d.size[0];
        }
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    if (_is_same_shape(inputs[0], outputs[0]->attr.size, outputs[0]->attr.dim_num))
    {
        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else if (VSI_NN_INTERPOLATION_BILINEAR == self->nn_param.resize_1d.type)
    {
        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESIZE_1D_BILINEAR_INTERNAL, 0, 0 );
        curr->node->nn_param.resize_1d_bilinear_internal.align_corners = self->nn_param.resize_1d.align_corners;
        curr->node->nn_param.resize_1d_bilinear_internal.factor = self->nn_param.resize_1d.factor;
        curr->node->nn_param.resize_1d_bilinear_internal.half_pixel_centers = \
                                              self->nn_param.resize_1d.half_pixel_centers;
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else if (VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR == self->nn_param.resize_1d.type)
    {
        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESIZE_1D_NEAREST_INTERNAL, 0, 0 );
        curr->node->nn_param.resize_1d_nearest_internal.align_corners = self->nn_param.resize_1d.align_corners;
        curr->node->nn_param.resize_1d_nearest_internal.factor = self->nn_param.resize_1d.factor;
        curr->node->nn_param.resize_1d_nearest_internal.half_pixel_centers = \
                                              self->nn_param.resize_1d.half_pixel_centers;
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_internal_deinit_node_wksp(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RESIZE_1D,
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


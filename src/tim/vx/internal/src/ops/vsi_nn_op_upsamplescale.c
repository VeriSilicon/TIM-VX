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
#include "utils/vsi_nn_constraint_check.h"

typedef struct _upsamplescale_local_data_t {
    int32_t placeholder;
} upsamplescale_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

#define _EPSILON 1e-8

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    int32_t stride = self->nn_param.upsamplescale.stride;
    float   scale  = self->nn_param.upsamplescale.scale;
    vsi_nn_kernel_param_t * param = NULL;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }

    if (stride == 1 || vsi_nn_abs(scale - 1.0f) == _EPSILON)
    {
        return vsi_nn_internal_compute_node( self );
    }

    param =vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "stride", stride );
    vsi_nn_kernel_param_add_float32( param, "scale", scale );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "upsamplescale",
        inputs, 1,
        outputs, 1, param );

    vsi_nn_kernel_param_release( &param );

    if( self->n )
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
    BEGIN_IO_TYPE_DECL(UPSAMPLESCALE, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_U8)
        IO_TYPE(D_F16,  D_I16)
        IO_TYPE(D_F16,  D_I8)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_U8,   D_U8)
        IO_TYPE(D_U8,   D_F16)
        IO_TYPE(D_I8,   D_I8)
        IO_TYPE(D_I8,   D_F16)
        IO_TYPE(D_I16,  D_I16)
        IO_TYPE(D_I16,  D_F16)
    END_IO_TYPE_DECL(UPSAMPLESCALE)
    if (!VALIDATE_OP_IO_TYPES(UPSAMPLESCALE, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    int32_t stride = self->nn_param.upsamplescale.stride;
    float scale = self->nn_param.upsamplescale.scale;

    if (stride == 1 && vsi_nn_abs(scale - 1.0f) == _EPSILON)
    {
        return vsi_nn_internal_optimize_node( self, direction );
    }
    else
    {
        return VSI_SUCCESS;
    }
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    int32_t stride = self->nn_param.upsamplescale.stride;
    float scale = self->nn_param.upsamplescale.scale;
    int32_t i = 0;
    vsi_nn_internal_node_t* curr = NULL;

    vsi_nn_internal_init_node_wksp(self);

    if (stride == 1 && vsi_nn_abs(scale - 1.0f) == _EPSILON)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_DATACONVERT, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];

        vsi_nn_internal_setup_node(self, curr);
    }
    else if (stride == 1)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_LINEAR, 0, 0);
        curr->node->nn_param.linear.a = scale;
        curr->node->nn_param.linear.b = 0;
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];

        vsi_nn_internal_setup_node(self, curr);
    }
    else if (vsi_nn_abs(scale - 1.0f) == _EPSILON)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RESIZE, 0, 0);
        curr->node->nn_param.resize.type = VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR;
        curr->node->nn_param.resize.align_corners = FALSE;
        curr->node->nn_param.resize.half_pixel_centers = FALSE;
        curr->node->nn_param.resize.size[0] = inputs[0]->attr.size[0] * stride;
        curr->node->nn_param.resize.size[1] = inputs[0]->attr.size[1] * stride;
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];

        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        outputs[0]->attr.size[0] = inputs[0]->attr.size[0] * stride;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1] * stride;
        for (i = 2; i < (int32_t)inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
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

    vsi_nn_internal_deinit_node_wksp( self );

    status = vsi_nn_op_common_deinit(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ UPSAMPLESCALE,
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


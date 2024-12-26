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
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _custom_letterbox_local_data_t {
    int32_t placeholder;
} custom_letterbox_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

int32_t my_round(float in)
{
    if (in >= 0)
    {
        return (int)(in + 0.5f);
    }
    else
    {
        return (int)(in - 0.5f);
    }
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_custom_letterbox_param * p;
    p = &(self->nn_param.custom_letterbox);
    int32_t shape_w = (int32_t)inputs[0]->attr.size[1];
    int32_t shape_h = (int32_t)inputs[0]->attr.size[2];
    int32_t new_shape_w = (int32_t)outputs[0]->attr.size[0];
    int32_t new_shape_h = (int32_t)outputs[0]->attr.size[1];
    vx_bool auto_bool = p->auto_bool;
    vx_bool scaleFill = p->scaleFill;
    vx_bool scaleup = p->scaleup;
    int32_t stride = p->stride;
    vx_bool center = p->center;

    float r = 1.0f;
    int32_t new_unpad_w = 0;
    int32_t new_unpad_h = 0;
    int32_t dw = 0;
    int32_t dh = 0;
    int32_t top = 0;
    int32_t bottom = 0;
    int32_t left = 0;
    int32_t right = 0;

    r = (float)fmin((float)new_shape_w / shape_w, (float)new_shape_h / shape_h);
    if (!scaleup)
    {
        r = (float)fmin(r, 1.0f);
    }

    new_unpad_w = my_round(r * shape_w);
    new_unpad_h = my_round(r * shape_h);
    dw = new_shape_w - new_unpad_w;
    dh = new_shape_h - new_unpad_h;
    if (auto_bool)
    {
        dw = dw % stride;
        dh = dh % stride;
    }
    else if (scaleFill)
    {
        dw = 0;
        dh = 0;
        new_unpad_w = new_shape_w;
        new_unpad_h = new_shape_h;
    }
    if (center)
    {
        top = my_round(dh / 2.0f - 0.1f);
        bottom = my_round(dh / 2.0f + 0.1f);
        left = my_round(dw / 2.0f - 0.1f);
        right = my_round(dw / 2.0f + 0.1f);
    }
    else
    {
        top = 0;
        bottom = my_round(dh + 0.1f);
        left = 0;
        right = my_round(dw + 0.1f);
    }

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "top", top);
    vsi_nn_kernel_param_add_int32( param, "bottom", bottom);
    vsi_nn_kernel_param_add_int32( param, "left", left);
    vsi_nn_kernel_param_add_int32( param, "right", right);
    vsi_nn_kernel_param_add_float32( param, "mean_r", p->mean_r);
    vsi_nn_kernel_param_add_float32( param, "mean_g", p->mean_g);
    vsi_nn_kernel_param_add_float32( param, "mean_b", p->mean_b);
    vsi_nn_kernel_param_add_float32( param, "scale_r", p->scale_r);
    vsi_nn_kernel_param_add_float32( param, "scale_g", p->scale_g);
    vsi_nn_kernel_param_add_float32( param, "scale_b", p->scale_b);
    vsi_nn_kernel_param_add_int32( param, "pad_value_r", p->pad_value_r);
    vsi_nn_kernel_param_add_int32( param, "pad_value_g", p->pad_value_g);
    vsi_nn_kernel_param_add_int32( param, "pad_value_b", p->pad_value_b);
    vsi_nn_kernel_param_add_int32( param, "reverse_channel", p->reverse_channel);

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
            "custom_letterbox",
            inputs, 1,
            outputs, 1, param );

    vsi_nn_kernel_param_release( &param );

    return VSI_SUCCESS;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(LETTERBOX, 1, 1)
        IO_TYPE(D_U8,         D_F16)
        IO_TYPE(D_U8,         D_U8|Q_ASYM)
        IO_TYPE(D_U8,         D_I8|Q_DFP)
        IO_TYPE(D_U8,         D_I8|Q_ASYM)
        IO_TYPE(D_U8,         D_I8|Q_SYM)
    END_IO_TYPE_DECL(LETTERBOX)
    if (!VALIDATE_OP_IO_TYPES(LETTERBOX, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
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
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = self->nn_param.custom_letterbox.new_shape_w;
        outputs[0]->attr.size[1] = self->nn_param.custom_letterbox.new_shape_h;
        outputs[0]->attr.size[2] = 3;
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_op_common_deinit(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CUSTOM_LETTERBOX,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

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
#include "kernel/vsi_nn_kernel.h"
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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_signalframe_param *p = &self->nn_param.signalframe;
    float pad_value = p->pad_value;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 33) == -1)
    {
        pad_value = (float)p->pad;
    }

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "frame_length",  p->window_length );
    vsi_nn_kernel_param_add_int32( param, "frame_step",  p->step );
    vsi_nn_kernel_param_add_int32( param, "axis",  p->axis );
    vsi_nn_kernel_param_add_int32( param, "pad_end",  p->pad_end );
    vsi_nn_kernel_param_add_float32( param, "pad_val",  pad_value );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "signal_frame",
        inputs, 1,
        outputs, 1, param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(SIGNAL_FRAME, 1, 1)
        IO_TYPE(D_F16,        D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_BF16,       D_BF16)
        IO_TYPE(D_F32,        D_F32)
    END_IO_TYPE_DECL(SIGNAL_FRAME)
    if (!VALIDATE_OP_IO_TYPES(SIGNAL_FRAME, self, inputs, self->input.num, outputs, self->output.num)) {
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
    uint32_t i = 0;
    vsi_bool ret = 0;
    uint32_t axis = 0;
    uint32_t num_frames = 0;
    uint32_t frame_axis = 0;
    uint32_t frame_step = 0;
    uint32_t frame_length = 0;
    vsi_nn_signalframe_param *p = &self->nn_param.signalframe;

    ret = TRUE;
    if( VSI_NN_DIM_AUTO != outputs[0]->attr.dim_num )
    {
        return ret;
    }

    axis = p->axis;
    if(axis >= inputs[0]->attr.dim_num)
    {
        return FALSE;
    }

    /* signal frame will increase dim num */
    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num + 1;
    for (i = 0; i < axis; i++)
    {
        outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
    }

    frame_step = p->step;
    frame_length = p->window_length;
    frame_axis = inputs[0]->attr.size[axis];
    num_frames = p->pad_end ?
        (frame_axis + frame_step - 1) / frame_step : (frame_axis - frame_length ) / frame_step + 1;

    outputs[0]->attr.size[axis] = frame_length;
    outputs[0]->attr.size[axis + 1] = num_frames;

    for (i = axis + 1; i < inputs[0]->attr.dim_num; i++)
    {
        outputs[0]->attr.size[i + 1] = inputs[0]->attr.size[i];
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SIGNAL_FRAME,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

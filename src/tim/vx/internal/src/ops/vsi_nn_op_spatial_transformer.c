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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (2)
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
    vsi_nn_spatial_transformer_param * p;
    p = (vsi_nn_spatial_transformer_param *)&self->nn_param.spatial_transformer;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "has_theta_1_1",  p->has_theta_1_1 );
    vsi_nn_kernel_param_add_int32( param, "has_theta_1_2",  p->has_theta_1_2 );
    vsi_nn_kernel_param_add_int32( param, "has_theta_1_3",  p->has_theta_1_3 );
    vsi_nn_kernel_param_add_int32( param, "has_theta_2_1",  p->has_theta_2_1 );
    vsi_nn_kernel_param_add_int32( param, "has_theta_2_2",  p->has_theta_2_2 );
    vsi_nn_kernel_param_add_int32( param, "has_theta_2_3",  p->has_theta_2_3 );
    vsi_nn_kernel_param_add_float32( param, "theta_1_1",  p->theta_1_1 );
    vsi_nn_kernel_param_add_float32( param, "theta_1_2",  p->theta_1_2 );
    vsi_nn_kernel_param_add_float32( param, "theta_1_3",  p->theta_1_3 );
    vsi_nn_kernel_param_add_float32( param, "theta_2_1",  p->theta_2_1 );
    vsi_nn_kernel_param_add_float32( param, "theta_2_2",  p->theta_2_2 );
    vsi_nn_kernel_param_add_float32( param, "theta_2_3",  p->theta_2_3 );
    vsi_nn_kernel_param_add_int32( param, "align_corners",  p->align_corners );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "spatial_transformer",
        inputs, 2,
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
    BEGIN_IO_TYPE_DECL(SPATIAL_TRANSFORMER, 2, 1)
        /* IO_TYPE(INPUT, OUTPUT) */
        IO_TYPE(D_F16,       D_F16,       D_F16)
        IO_TYPE(D_F16,       D_F16,       D_I16|Q_DFP)
        IO_TYPE(D_F16,       D_F16,       D_U8|Q_ASYM)
        IO_TYPE(D_F16,       D_F16,       D_I8|Q_DFP)
        IO_TYPE(D_F16,       D_I16|Q_DFP, D_F16)
        IO_TYPE(D_F16,       D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_F16,       D_I8|Q_DFP,  D_F16)
        IO_TYPE(D_I16|Q_DFP, D_F16,       D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_F16,       D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_F16,       D_F16)
        IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_F16)
        IO_TYPE(D_F16,       D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_F16,       D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_F16,       D_I8|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_F16,       D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_F16,       D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_F16,       D_I8|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP, D_F16,       D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_F16,       D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,  D_F16,       D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I8|Q_DFP)
    END_IO_TYPE_DECL(SPATIAL_TRANSFORMER)
    if (!VALIDATE_OP_IO_TYPES(SPATIAL_TRANSFORMER, self, inputs, self->input.num, outputs, self->output.num))
    {
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
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_spatial_transformer_param * p;
    p = (vsi_nn_spatial_transformer_param *)&node->nn_param.spatial_transformer;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t i = 0;

        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = p->output_W;
        outputs[0]->attr.size[1] = p->output_H;
        if (p->output_W == 0)
        {
            outputs[0]->attr.size[0] = inputs[0]->attr.size[0];
        }

        if (p->output_H == 0)
        {
            outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
        }

        for (i = 2; i < outputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.spatial_transformer.align_corners = FALSE;

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SPATIAL_TRANSFORMER,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

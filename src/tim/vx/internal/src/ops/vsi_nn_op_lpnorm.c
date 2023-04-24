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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _lpnorm_local_data_t {
    int32_t placeholder;
} lpnorm_local_data_t;

/*
 Declare number of input and output.
 */
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
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    uint32_t new_rank = 0;
    int32_t  new_axis = 0;
    vsi_size_t shapes[1][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t p = (int32_t)self->nn_param.lpnorm.p;
    int32_t axis = (int32_t)self->nn_param.lpnorm.axis;
    int32_t dim = (int32_t)inputs[0]->attr.dim_num;

    if (axis == -1) axis = dim - 1;
    vsi_nn_kernel_optimize_softmax_shape(inputs[0]->attr.size,
                                         inputs[0]->attr.dim_num,
                                         axis,
                                         shapes[0],
                                         &new_rank,
                                         &new_axis);

    reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
            inputs[0], shapes[0], new_rank );
    reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], shapes[0], new_rank );

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32(param, "p", p);

    if (p == 1)
    {
        vsi_nn_kernel_param_add_int32(param, "axis", new_axis);
        self->n = (vx_node)vsi_nn_kernel_selector(self->graph,"l1norm",
            &reshape_tensors[0], _INPUT_NUM, &reshape_tensors[1], _OUTPUT_NUM, param);
    }
    else
    {
        vsi_nn_kernel_param_add_int32(param, "axis", axis);
        self->n = (vx_node)vsi_nn_kernel_selector(self->graph,"l2_norm",
            inputs, _INPUT_NUM, outputs, _OUTPUT_NUM, param);
    }

    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release(&param);
    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(LPNORM, 1, 1)
        IO_TYPE(D_F32,   D_F32)
        IO_TYPE(D_F16,   D_F16)
        IO_TYPE(D_I16|Q_SYM,   D_I16|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,   D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,    D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,   D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,    D_I8|Q_DFP)
        IO_TYPE(D_F32,   D_I16|Q_SYM)
        IO_TYPE(D_F16,   D_I16|Q_SYM)
        IO_TYPE(D_F32,   D_I16|Q_DFP)
        IO_TYPE(D_F16,   D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_SYM,   D_F32)
        IO_TYPE(D_U8|Q_ASYM,   D_F32)
        IO_TYPE(D_I8|Q_SYM,    D_F32)
        IO_TYPE(D_I16|Q_DFP,   D_F32)
        IO_TYPE(D_I8|Q_DFP,    D_F32)
        IO_TYPE(D_F32,   D_U8|Q_ASYM)
        IO_TYPE(D_F16,   D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,   D_F16)
        IO_TYPE(D_U8|Q_ASYM,   D_F16)
        IO_TYPE(D_I8|Q_SYM,    D_F16)
        IO_TYPE(D_I16|Q_DFP,   D_F16)
        IO_TYPE(D_I8|Q_DFP,    D_F16)
        IO_TYPE(D_F32,   D_I8|Q_SYM)
        IO_TYPE(D_F16,   D_I8|Q_SYM)
        IO_TYPE(D_F32,   D_I8|Q_DFP)
        IO_TYPE(D_F16,   D_I8|Q_DFP)
    END_IO_TYPE_DECL(LPNORM)

    if (!VALIDATE_OP_IO_TYPES(
            LPNORM, self, inputs, self->input.num, outputs, self->output.num))
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
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_size_t i = 0;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        for (i = 0; i < outputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    self->nn_param.lpnorm.p = 2;
    self->nn_param.lpnorm.axis = -1;
    return VSI_SUCCESS;
} /* op_init() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LPNORM,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS


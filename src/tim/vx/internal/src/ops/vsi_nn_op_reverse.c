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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

static vsi_bool _is_same_quant
    (
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_dtype_t *src_dtype = NULL,*dst_dtype = NULL;

    src_dtype = &inputs[0]->attr.dtype;
    dst_dtype = &outputs[0]->attr.dtype;

    if (vsi_nn_DtypeCompare(src_dtype, dst_dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;

    vx_nn_tensor_reverse_params_t para;
    vsi_nn_reverse_param * p;
    int32_t axes[VSI_NN_MAX_DIM_NUM] = {0};

    if ( _is_same_quant(inputs, outputs) )
    {
        p = &self->nn_param.reverse;
        memcpy(axes, p->axis, sizeof(int32_t) * p->axis_num);
        para.axis = axes;
        para.numberOfAxis = p->axis_num;
        self->n = vxTensorReverse( self->graph->g, inputs[0]->t, &para,
            sizeof(vx_nn_tensor_reverse_params_t), outputs[0]->t );
        if( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }

        return status;
    }
    else
    {
        return vsi_nn_internal_compute_node( self );
    }
} /* op_compute() */


static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REVERSE, 1, 1)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_U8|Q_DFP,   D_U8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I32|Q_DFP,  D_I32|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,  D_I8|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM, D_I16|Q_ASYM)
        IO_TYPE(D_I32|Q_ASYM, D_I32|Q_ASYM)
        IO_TYPE(D_U8|Q_SYM_PC,   D_U8|Q_SYM_PC)
        IO_TYPE(D_I8|Q_SYM_PC,   D_I8|Q_SYM_PC)
        IO_TYPE(D_I16|Q_SYM_PC,  D_I16|Q_SYM_PC)
        IO_TYPE(D_I32|Q_SYM_PC,  D_I32|Q_SYM_PC)
        IO_TYPE(D_U8,   D_U8)
        IO_TYPE(D_I8,   D_I8)
        IO_TYPE(D_I16,  D_I16)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_F32,  D_BF16)
        IO_TYPE(D_BF16, D_F32)
        IO_TYPE(D_F16,        D_I32)
        IO_TYPE(D_U8|Q_ASYM,  D_I32)
        IO_TYPE(D_I8|Q_DFP,   D_I32)
        IO_TYPE(D_I16|Q_DFP,  D_I32)

        /* HW 9.0 */
        IO_TYPE(D_BF16, D_BF16)
    END_IO_TYPE_DECL(REVERSE)
    if(!VALIDATE_OP_IO_TYPES(REVERSE, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }
    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_internal_deinit_node_wksp(self);
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if ( _is_same_quant(inputs, outputs) )
    {
        return VSI_SUCCESS;
    }
    else
    {
        return vsi_nn_internal_optimize_node(self, direction );
    }
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_internal_node_t* curr = NULL;

    vsi_nn_internal_init_node_wksp(self);

    ret = vsi_nn_op_common_setup(self, inputs, outputs);

    if ( _is_same_quant(inputs, outputs) == FALSE )
    {
        vsi_nn_internal_tensor_t* output_tensor = NULL;
        vsi_nn_tensor_attr_t attr;
        int32_t size = sizeof( attr.size );

        memcpy( &attr, &inputs[0]->attr, sizeof( attr ) );
        memcpy( &attr.size, &outputs[0]->attr.size, size );
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_REVERSE, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = output_tensor->t;
        curr->node->nn_param.reverse.axis = self->nn_param.reverse.axis;
        curr->node->nn_param.reverse.axis_num = self->nn_param.reverse.axis_num;
        vsi_nn_internal_setup_node(self, curr);

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_DATACONVERT, 0, 0);
        curr->inputs[0] = output_tensor->t;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REVERSE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

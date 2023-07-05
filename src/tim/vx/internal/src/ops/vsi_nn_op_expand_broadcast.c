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
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_error.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_compute_node( self );
}

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(EXPAND_BROADCAST, 1, 1)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_F32,          D_F16)
        IO_TYPE(D_F16,          D_F32)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_F16,          D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_I16|Q_ASYM)
        IO_TYPE(D_F16,          D_I16|Q_SYM)
        IO_TYPE(D_F16,          D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_I8|Q_ASYM)
        IO_TYPE(D_F16,          D_I8|Q_SYM)
        IO_TYPE(D_F16,          D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_F16)
        IO_TYPE(D_I16|Q_ASYM,   D_F16)
        IO_TYPE(D_I16|Q_SYM,    D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,    D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,     D_F16)
        IO_TYPE(D_I8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_SYM,     D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_F32,          D_BF16)
        IO_TYPE(D_BF16,         D_F32)
        IO_TYPE(D_I32|Q_DFP,    D_I32|Q_DFP)
        IO_TYPE(D_I32|Q_ASYM,   D_I32|Q_ASYM)
        IO_TYPE(D_BOOL8,        D_BOOL8)
    END_IO_TYPE_DECL(EXPAND_BROADCAST)
    if (!VALIDATE_OP_IO_TYPES(EXPAND_BROADCAST, self, inputs, self->input.num, outputs, self->output.num))
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
    uint32_t i;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* input_0 = NULL;
    vsi_nn_internal_tensor_t *input_1 = NULL;
    vsi_nn_internal_tensor_t* input_2 = NULL;
    vsi_nn_internal_node_t* mul_node = NULL;
    vsi_nn_tensor_t* mul_input = NULL;
    int32_t use_virtual_tensor = 1;
    vsi_bool is_same_shape = TRUE;
    vsi_nn_expand_broadcast_param *p = &self->nn_param.expand_broadcast;

    vsi_nn_internal_init_node_wksp(self);

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = p->dim_num;
    if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_NONE &&
        (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32 ||
        inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16))
    {
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    }
    else if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_BOOL8)
    {
        attr.dtype.vx_type = VSI_NN_TYPE_BOOL8;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = TRUE;
    for (i = 0; i < p->dim_num; i++)
    {
        vsi_size_t sz = i < inputs[0]->attr.dim_num ?
            inputs[0]->attr.size[i] : 1;

        attr.size[i] = p->shape[i];
        if (( p->shape[i] != sz && p->shape[i] != 1)
            && is_same_shape)
        {
            is_same_shape = FALSE;
        }
    }
    input_1 = vsi_nn_internal_new_tensor( self, &attr, 1.0f );
    CHECK_PTR_FAIL_GOTO(input_1, "Create tensor failed", final);

    if (p->dimensions_num > 0)
    {
        vsi_nn_internal_node_t* reshape_node = NULL;
        vsi_size_t* reshape_input_size = NULL;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        vsi_nn_internal_init_tensor_attr(&attr, &inputs[0]->attr.dtype, use_virtual_tensor);
        input_0 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        CHECK_PTR_FAIL_GOTO(input_0, "Create internal tensor failed", final);
        reshape_node = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
        CHECK_PTR_FAIL_GOTO(reshape_node, "Create internal node failed", final);
        reshape_input_size = (vsi_size_t*)vsi_nn_internal_new_node_param(reshape_node,
            VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        CHECK_PTR_FAIL_GOTO_RLS_INTERNAL_NODE(reshape_input_size, reshape_node,
            "Create internal buffer failed", final);
        for (i = 0; i < p->dim_num; i++)
        {
            reshape_input_size[i] = 1;
        }
        for (i = 0; i < p->dimensions_num; i++)
        {
            reshape_input_size[p->dimensions[i]] = p->shape[p->dimensions[i]];
        }

        reshape_node->node->nn_param.reshape2.size = reshape_input_size;
        reshape_node->node->nn_param.reshape2.dim_num = p->dim_num;
        reshape_node->inputs[0] = inputs[0];
        reshape_node->outputs[0] = input_0->t;
        vsi_nn_internal_setup_node( self, reshape_node );
        mul_input = input_0->t;
    }
    else
    {
        mul_input = inputs[0];
    }

    if (is_same_shape)
    {
        vsi_nn_internal_node_t* curr = NULL;
        vsi_nn_tensor_t* temp_tensor = NULL;

        if (input_1->t->attr.dim_num != mul_input->attr.dim_num)
        {
            vsi_size_t* shape_sizes = NULL;
            uint32_t rank0 = input_1->t->attr.dim_num;
            uint32_t rank1 = mul_input->attr.dim_num;
            uint32_t rank = vsi_nn_max( rank0, rank1 );

            memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
            vsi_nn_internal_init_tensor_attr(&attr, &mul_input->attr.dtype, use_virtual_tensor);
            input_2 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            CHECK_PTR_FAIL_GOTO(input_2, "Create internal tensor failed", final);

            curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RESHAPE2, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
            shape_sizes = (vsi_size_t*)vsi_nn_internal_new_node_param(curr,
                VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
            CHECK_PTR_FAIL_GOTO_RLS_INTERNAL_NODE(shape_sizes, curr,
                "Create internal buffer failed", final);
            for (i = 0; i < rank; i++)
            {
                shape_sizes[i] = i < rank1 ? mul_input->attr.size[i] : 1;
            }
            curr->node->nn_param.reshape2.size = shape_sizes;
            curr->node->nn_param.reshape2.dim_num = rank;
            curr->inputs[0] = mul_input;
            curr->outputs[0] = input_2->t;
            vsi_nn_internal_setup_node( self, curr );

            temp_tensor = input_2->t;
        }
        else
        {
            temp_tensor = mul_input;
        }

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_DATACONVERT, 0, 0 );
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->inputs[0] = temp_tensor;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        mul_node = vsi_nn_internal_new_node(self, VSI_NN_OP_MULTIPLY, 0, 0 );
        CHECK_PTR_FAIL_GOTO(mul_node, "Create internal node failed", final);
        mul_node->inputs[0] = mul_input;
        mul_node->inputs[1] = input_1->t;
        mul_node->node->nn_param.multiply.scale = 1.0f;
        mul_node->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        mul_node->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
        mul_node->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, mul_node);
    }

    return TRUE;
final:
    return FALSE;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ EXPAND_BROADCAST,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

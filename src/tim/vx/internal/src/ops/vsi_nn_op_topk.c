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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "vsi_nn_error.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (2)

vsi_nn_tensor_t* _create_permute_node
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    uint32_t* perm,
    uint32_t dim_num,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_t* tensor0 = NULL;
    vsi_nn_tensor_t *output = NULL;

    if (output_tensor)
    {
        output = output_tensor;
    }
    else
    {
        uint32_t i = 0;
        vsi_nn_tensor_attr_t attr;
        memcpy(&attr, &input_tensor->attr, sizeof(attr));
        attr.vtl = use_virtual_tensor;
        for ( i = 0; i < dim_num; i++ )
        {
            attr.size[i] = input_tensor->attr.size[perm[i]];
        }
        tensor0 = vsi_nn_CreateTensor( self->graph, &attr );
        CHECK_PTR_FAIL_GOTO( tensor0, "Create tensor fail.", final );
        output = tensor0;
    }
    self->n = vxTensorPermuteNode(
        self->graph->g,
        input_tensor->t,
        output->t,
        perm,
        dim_num
        );
    if (self->n == NULL)
    {
        vsi_safe_release_tensor(tensor0);
    }

final:
    return tensor0;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    int32_t new_axis0 = 0;
    int32_t new_axis1 = 0;
    int32_t axis = self->nn_param.topk.axis;
    int32_t top_k = self->nn_param.topk.k;
    vsi_nn_tensor_t * in_tensor = NULL;
    vsi_nn_tensor_t * out0_tensor = NULL;
    vsi_nn_tensor_t * out1_tensor = NULL;
    vsi_bool ret = FALSE;

    ret = vsi_nn_kernel_optimize_softmax_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num, axis,
            shapes[0], &rank_in, &new_axis0);

    ret = vsi_nn_kernel_optimize_softmax_shape(
            outputs[0]->attr.size, outputs[0]->attr.dim_num, axis,
            shapes[1], &rank_out, &new_axis1);

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "top_k", top_k );

    if (ret)
    {
        uint32_t perm_in[VSI_NN_MAX_DIM_NUM] = {0};
        uint32_t perm_out[VSI_NN_MAX_DIM_NUM] = {0};
        vsi_nn_tensor_t* input_tensor = NULL;
        vsi_nn_tensor_t* outputs_tensor[2] = {NULL};

        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[1], rank_in );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[1], shapes[1], rank_in );

        axis = new_axis0;

        if (axis != 0)
        {
            uint32_t i = 0;
            uint32_t index = 0;

            vsi_nn_tensor_attr_t attr0, attr1;
            memcpy(&attr0, &reshape_tensors[1]->attr, sizeof(attr0));
            memcpy(&attr1, &reshape_tensors[2]->attr, sizeof(attr1));

            attr0.vtl = TRUE;
            attr1.vtl = TRUE;
            attr0.size[index] = (vsi_size_t)top_k;
            attr1.size[index] = (vsi_size_t)top_k;
            perm_in[index ++] = (uint32_t)axis;
            for ( i = 0; i < rank_in; i++ )
            {
                if ((int32_t)i == axis)
                    continue;
                attr0.size[index] = shapes[1][i];
                attr1.size[index] = shapes[1][i];
                perm_in[index ++] = i;
            }

            perm_out[axis] = 0;
            for ( i = 1, index = 0; i < rank_in; i++ )
            {
                if ((int32_t)index == axis)
                {
                    index ++;
                }
                perm_out[index ++] = i;
            }

            out0_tensor = vsi_nn_CreateTensor( self->graph, &attr0 );
            CHECK_PTR_FAIL_GOTO( out0_tensor, "Create tensor fail.", final );
            out1_tensor = vsi_nn_CreateTensor( self->graph, &attr1 );
            CHECK_PTR_FAIL_GOTO( out1_tensor, "Create tensor fail.", final );

            in_tensor = _create_permute_node(self, reshape_tensors[0], NULL, perm_in, rank_in, TRUE);
            CHECK_PTR_FAIL_GOTO( in_tensor, "Create internal tensor fail.", final );

            input_tensor = in_tensor;
            outputs_tensor[0] = out0_tensor;
            outputs_tensor[1] = out1_tensor;
        }
        else
        {
            input_tensor = reshape_tensors[0];
            outputs_tensor[0] = reshape_tensors[1];
            outputs_tensor[1] = reshape_tensors[2];
        }

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "topk",
                &input_tensor, _INPUT_NUM,
                outputs_tensor, _OUTPUT_NUM, param );

        if (axis != 0)
        {
            _create_permute_node(self, outputs_tensor[0], reshape_tensors[1], perm_out, rank_in, TRUE);
            _create_permute_node(self, outputs_tensor[1], reshape_tensors[2], perm_out, rank_in, TRUE);
        }
    }

    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

final:
    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );
    vsi_safe_release_tensor( reshape_tensors[2] );
    vsi_safe_release_tensor( in_tensor );
    vsi_safe_release_tensor( out0_tensor );
    vsi_safe_release_tensor( out1_tensor );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(TOPK, _INPUT_NUM, _OUTPUT_NUM)
        IO_TYPE(D_F16,        D_F16,        D_I32)
        IO_TYPE(D_F16,        D_U8|Q_ASYM,  D_I32)
        IO_TYPE(D_F16,        D_I16|Q_DFP,  D_I32)
        IO_TYPE(D_F32,        D_F32,        D_I32)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP,   D_I32)
        IO_TYPE(D_I8|Q_ASYM,  D_I8|Q_ASYM,  D_I32)
        IO_TYPE(D_I8|Q_SYM,   D_I8|Q_SYM,   D_I32)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_I32)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I32)
        IO_TYPE(D_I16|Q_ASYM, D_I16|Q_ASYM, D_I32)
        IO_TYPE(D_I16|Q_SYM,  D_I16|Q_SYM,  D_I32)
        IO_TYPE(D_I32,        D_I32,        D_I32)
    END_IO_TYPE_DECL(TOPK)
    if (!VALIDATE_OP_IO_TYPES(TOPK, self, inputs, self->input.num, outputs, self->output.num))
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
    /* TODO: Add code to comput outputs' shape. */
    uint32_t i;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        vsi_nn_topk_param * p;

        p = &(self->nn_param.topk);

        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[p->axis] = p->k;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            if ((int32_t)i == p->axis)
            {
                continue;
            }
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
    }

    if ( VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num )
    {
        vsi_nn_topk_param * p;

        p = &(self->nn_param.topk);

        outputs[1]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[1]->attr.size[p->axis] = p->k;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            if ((int32_t)i == p->axis)
            {
                continue;
            }
            outputs[1]->attr.size[i] = inputs[0]->attr.size[i];
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
    self->nn_param.topk.axis = 0;

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ TOPK,
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

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
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _mod_local_data_t {
    int32_t placeholder;
} mod_local_data_t;

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
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_size_t new_rank = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;
    int32_t isfmod = 0;

    if (NULL == self)
    {
        return VSI_FAILURE;
    }

    isfmod = (int32_t)self->nn_param.mod.fmod;

    param = vsi_nn_kernel_param_create();

    ret = vsi_nn_kernel_optimize_eltwise_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            inputs[1]->attr.size, inputs[1]->attr.dim_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    vsi_nn_kernel_param_add_int32( param, "isfmod",  isfmod );

    if (ret)
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], shapes[1], new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[2], new_rank );
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "mod",
                                                 &reshape_tensors[0], _INPUT_NUM,
                                                 &reshape_tensors[2], _OUTPUT_NUM, param );
        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        vsi_nn_ReleaseTensor( &reshape_tensors[2] );
    }

    if (self->n)
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
    BEGIN_IO_TYPE_DECL(MOD, 2, 1)
        IO_TYPE(D_F16,          D_F16,          D_U8|Q_ASYM)
        IO_TYPE(D_F16,          D_F16,          D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_F16,          D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_F16,          D_F16)
        IO_TYPE(D_F32,          D_F32,          D_F32)
        IO_TYPE(D_I32,          D_I32,          D_I32)
        IO_TYPE(D_I32,          D_I32,          D_U8|Q_ASYM)
        IO_TYPE(D_BF16,         D_BF16,         D_BF16)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP,     D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP,    D_F16)
        IO_TYPE(D_U8|Q_ASYM,    D_I32,          D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_I32,          D_U8|Q_ASYM)
        IO_TYPE(D_I32,          D_I16|Q_DFP,    D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP,    D_U8|Q_ASYM)
        IO_TYPE(D_I16,          D_I32,          D_I32)
        IO_TYPE(D_I8|Q_DFP,     D_I32,          D_I8|Q_DFP)
        IO_TYPE(D_I32,          D_I32,          D_I8|Q_DFP)
        IO_TYPE(D_I32,          D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,    D_I32,          D_I16|Q_DFP)
        IO_TYPE(D_I32,          D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I32,          D_I32,          D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_F16,          D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM,    D_F16)
        IO_TYPE(D_F16,          D_F16,          D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM,     D_F16)
        IO_TYPE(D_I8|Q_ASYM,    D_I32,          D_I8|Q_ASYM)
        IO_TYPE(D_I32,          D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I32,          D_I32,          D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I32,          D_I8|Q_SYM)
        IO_TYPE(D_I32,          D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_I32,          D_I32,          D_I8|Q_SYM)
        IO_TYPE(D_F16,          D_F16,          D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM,   D_F16)
        IO_TYPE(D_F16,          D_F16,          D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM,    D_F16)
        IO_TYPE(D_I16|Q_ASYM,   D_I32,          D_I16|Q_ASYM)
        IO_TYPE(D_I32,          D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I32,          D_I32,          D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I32,          D_I16|Q_SYM)
        IO_TYPE(D_I32,          D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_I32,          D_I32,          D_I16|Q_SYM)
    END_IO_TYPE_DECL(MOD)
    if (!VALIDATE_OP_IO_TYPES(MOD, self, inputs, self->input.num, outputs, self->output.num))
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
    uint32_t i, out_rank, in1_rank, in2_rank;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = TRUE;

    VSI_UNREFERENCED(self);

    in1_rank = inputs[0]->attr.dim_num;
    in2_rank = inputs[1]->attr.dim_num;
    out_rank = vsi_nn_max( in1_rank, in2_rank );

    for(i = 0; i < out_rank; i++)
    {
        vsi_size_t sz0, sz1;
        sz0 = i < in1_rank ? inputs[0]->attr.size[i] : 1;
        sz1 = i < in2_rank ? inputs[1]->attr.size[i] : 1;
        shape[i] = vsi_nn_max( sz0, sz1 );
    }

    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        outputs[0]->attr.dim_num = out_rank;
        memcpy( outputs[0]->attr.size, shape, out_rank * sizeof(vsi_size_t) );
    }
    else
    {
        vsi_size_t total_size_got;
        vsi_size_t total_size_expected;
        total_size_expected = vsi_nn_ShapeProduct( shape, out_rank );
        total_size_got = vsi_nn_ShapeProduct( outputs[0]->attr.size,
                outputs[0]->attr.dim_num );
        if (total_size_expected != total_size_got)
        {
            VSILOGW("Output size mismatch, expect %"VSI_SIZE_T_SPECIFIER", but got %"VSI_SIZE_T_SPECIFIER"",
                    total_size_expected, total_size_got);
            ret = FALSE;
        }
    }

    return ret;
} /* op_setup() */

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MOD,
    /* init       */ NULL,
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


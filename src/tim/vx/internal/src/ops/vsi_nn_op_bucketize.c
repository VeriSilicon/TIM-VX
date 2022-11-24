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
#include "utils/vsi_nn_util.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _bucketize_local_data_t {
    int32_t placeholder;
} bucketize_local_data_t;

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
    int32_t right = self->nn_param.bucketize.right;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t new_rank = 0;
    vsi_bool ret = TRUE;
    vsi_nn_kernel_param_t * param = NULL;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "right",  right );

    ret = vsi_nn_kernel_optimize_element_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );

    if ( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shape, new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shape, new_rank );
        shape[0] = inputs[1]->attr.size[0];
        shape[1] = 1;
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], shape, 2 );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "bucketize",
                &reshape_tensors[0], 2,
                &reshape_tensors[2], 1, param );

        vsi_safe_release_tensor( reshape_tensors[0] );
        vsi_safe_release_tensor( reshape_tensors[1] );
        vsi_safe_release_tensor( reshape_tensors[2] );
    }

    if ( self->n )
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
    BEGIN_IO_TYPE_DECL(BUCKETIZE, 2, 1)
        IO_TYPE(D_U32,          D_U32,          D_I32)
        IO_TYPE(D_I32,          D_I32,          D_I32)
        IO_TYPE(D_F32,          D_F32,          D_I32)
        IO_TYPE(D_F16,          D_F16,          D_I32)
        IO_TYPE(D_BF16,         D_BF16,         D_I32)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM,   D_I32)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM,    D_I32)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP,    D_I32)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM,    D_I32)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM,    D_I32)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM,     D_I32)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP,     D_I32)
    END_IO_TYPE_DECL(BUCKETIZE)
    if (!VALIDATE_OP_IO_TYPES(BUCKETIZE, self, inputs, self->input.num, outputs, self->output.num))
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
    uint32_t i, out_rank;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = TRUE;

    out_rank = inputs[0]->attr.dim_num;

    for (i = 0; i < out_rank; i++)
    {
        shape[i] = inputs[0]->attr.size[i];
    }
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
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
        if ( total_size_expected != total_size_got )
        {
            VSILOGW("Output size mismatch, expect %"VSI_SIZE_T_SPECIFIER", but got %"VSI_SIZE_T_SPECIFIER"",
                    total_size_expected, total_size_got);
            ret = FALSE;
        }
    }

    return ret;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    self->nn_param.bucketize.right = FALSE;

    return VSI_SUCCESS;
} /* op_init() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BUCKETIZE,
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


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
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status _comparisons_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    vsi_size_t new_rank = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_relational_ops_type_t op_type = self->nn_param.relational_ops.op;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;

    // TODO: This optimzie is a hack for gpu path,
    // it should be moved to gpu kernel setup.
    ret = vsi_nn_kernel_optimize_eltwise_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            inputs[1]->attr.size, inputs[1]->attr.dim_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );
    if( ret )
    {
        // Add params
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], shapes[1], new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[2], new_rank );

        if (shapes[1][3] > shapes[0][3] && new_rank == 4)
        {
            vsi_nn_tensor_t* reshape_tmp;
            reshape_tmp = reshape_tensors[0];
            reshape_tensors[0] = reshape_tensors[1];
            reshape_tensors[1] = reshape_tmp;
            if (VSI_NN_RELATIONAL_OPS_GREAT == op_type)
            {
                op_type = VSI_NN_RELATIONAL_OPS_LESS;
            }
            else if (VSI_NN_RELATIONAL_OPS_LESS == op_type)
            {
                op_type = VSI_NN_RELATIONAL_OPS_GREAT;
            }
            else if (VSI_NN_RELATIONAL_OPS_GREAT_EQUAL == op_type)
            {
                op_type = VSI_NN_RELATIONAL_OPS_LESS_EQUAL;
            }
            else if (VSI_NN_RELATIONAL_OPS_LESS_EQUAL == op_type)
            {
                op_type = VSI_NN_RELATIONAL_OPS_GREAT_EQUAL;
            }
        }

        param = vsi_nn_kernel_param_create();
        vsi_nn_kernel_param_add_int32( param, "operation", op_type );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                kernel_name,
                &reshape_tensors[0], 2,
                &reshape_tensors[2], 1, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        vsi_nn_ReleaseTensor( &reshape_tensors[2] );

        vsi_nn_kernel_param_release( &param );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
} /* _eltwise_op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(RELATIONAL_OPS, 2, 1)
        IO_TYPE(D_F16,  D_F16, D_BOOL8)
        IO_TYPE(D_F16,  D_I16|Q_DFP, D_BOOL8)
        IO_TYPE(D_F16,  D_I8|Q_DFP, D_BOOL8)
        IO_TYPE(D_F16,  D_U8|Q_ASYM, D_BOOL8)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP, D_BOOL8)
        IO_TYPE(D_I16|Q_DFP,  D_F16, D_BOOL8)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP, D_BOOL8)
        IO_TYPE(D_I8|Q_DFP,  D_F16, D_BOOL8)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM, D_BOOL8)
        IO_TYPE(D_U8|Q_ASYM,  D_F16, D_BOOL8)
        IO_TYPE(D_BF16,  D_BF16,  D_BOOL8)
        IO_TYPE(D_BOOL8, D_BOOL8, D_BOOL8)
        IO_TYPE(D_F32, D_F32, D_BOOL8)
        IO_TYPE(D_I32, D_I32, D_BOOL8)

        IO_TYPE(D_F16,  D_F16, D_I8)
        IO_TYPE(D_F16,  D_I16|Q_DFP, D_I8)
        IO_TYPE(D_F16,  D_I8|Q_DFP, D_I8)
        IO_TYPE(D_F16,  D_U8|Q_ASYM, D_I8)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP, D_I8)
        IO_TYPE(D_I16|Q_DFP,  D_F16, D_I8)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP, D_I8)
        IO_TYPE(D_I8|Q_DFP,  D_F16, D_I8)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM, D_I8)
        IO_TYPE(D_U8|Q_ASYM,  D_F16, D_I8)
        IO_TYPE(D_BF16,  D_BF16,  D_I8)
        IO_TYPE(D_BOOL8, D_BOOL8, D_I8)
        IO_TYPE(D_F32, D_F32, D_I8)
        IO_TYPE(D_I32, D_I32, D_I8)
    END_IO_TYPE_DECL(RELATIONAL_OPS)
    if(!VALIDATE_OP_IO_TYPES(RELATIONAL_OPS, self, inputs, self->input.num, outputs, self->output.num)) {
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
    vsi_size_t i, out_rank, in1_rank, in2_rank;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = TRUE;

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
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = (uint32_t)out_rank;
        memcpy( outputs[0]->attr.size, shape, out_rank * sizeof(vsi_size_t) );
    }
    else
    {
        vsi_size_t total_size_got;
        vsi_size_t total_size_expected;
        total_size_expected = vsi_nn_ShapeProduct( shape, out_rank );
        total_size_got = vsi_nn_ShapeProduct( outputs[0]->attr.size,
                outputs[0]->attr.dim_num );
        if( total_size_expected != total_size_got )
        {
            VSILOGW("Output size mismatch, expect %"VSI_SIZE_T_SPECIFIER", but got %"VSI_SIZE_T_SPECIFIER"",
                    total_size_expected, total_size_got);
            ret = FALSE;
        }
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_COMPARISONS_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _comparisons_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
DEF_OP_REG(name, NULL, op_compute_##kernel_name, vsi_nn_op_common_deinit, op_check, op_setup, NULL, 2, 1)

DEF_COMPARISONS_OP( RELATIONAL_OPS, relational_ops );


#undef DEF_COMPARISONS_OP

#ifdef __cplusplus
}
#endif

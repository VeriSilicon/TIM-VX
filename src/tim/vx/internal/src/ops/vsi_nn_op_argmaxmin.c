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
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status _argmaxmin_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    int32_t axis = 0;
    int32_t new_axis = 0;
    uint32_t axis_size = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;

    param =vsi_nn_kernel_param_create();
    if (strcmp(kernel_name, "argmax") == 0)
    {
        vsi_nn_argmax_param * p = &(self->nn_param.argmax);
        axis = p->axis;
    }
    else
    {
        vsi_nn_argmin_param * p = &(self->nn_param.argmin);
        axis = p->axis;
    }

    // TODO: This optimzie is a hack for gpu path,
    // it should be moved to gpu kernel setup.
    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            &axis, 1,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            &new_axis, &axis_size);

    // Add params
     vsi_nn_kernel_param_add_int32( param, "axis", new_axis );

    if( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[1], rank_out );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                kernel_name,
                &reshape_tensors[0], 1,
                &reshape_tensors[1], 1, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* _argmaxmin_op_compute() */

static vsi_bool _argmaxmin_op_setup
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t axis = 0;
    vsi_bool keep_dims = FALSE;
    vsi_bool ret = TRUE;

    if (strcmp(kernel_name, "argmax") == 0)
    {
        vsi_nn_argmax_param * p = &(self->nn_param.argmax);
        axis = p->axis;
        keep_dims = p->keep_dims;

        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            p->axis = axis;
        }
    }
    else
    {
        vsi_nn_argmin_param * p = &(self->nn_param.argmin);
        axis = p->axis;
        keep_dims = p->keep_dims;

        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            p->axis = axis;
        }
    }

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t i = 0;
        uint32_t i_rank = inputs[0]->attr.dim_num;
        uint32_t o_rank = keep_dims ? i_rank : i_rank - 1;
        int8_t   is_scalar = o_rank == 0;

        outputs[0]->attr.dim_num = is_scalar ? 1 : o_rank;
        vsi_nn_SetTensorIsScalar(outputs[0], is_scalar);

        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = is_scalar ? 1 : inputs[0]->attr.size[i];
        }

        if (keep_dims)
        {
            outputs[0]->attr.size[(uint32_t)axis] = 1;
        }
        else
        {
            for (i = axis; i < outputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = is_scalar ? 1 : inputs[0]->attr.size[i + 1];
            }
        }

        if (inputs[0]->attr.dim_num == 1)
        {
            outputs[0]->attr.dim_num = 1;
            outputs[0]->attr.size[0] = 1;
        }
    }

    return ret;
} /* _argmaxmin_op_setup() */


static vsi_status _argmaxmin_op_init
    (
    const char * kernel_name,
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    if (strcmp(kernel_name, "argmax") == 0)
    {
        vsi_nn_argmax_param* p = &(self->nn_param.argmax);
        p->axis = 2;
        p->keep_dims = FALSE;
    }
    else
    {
        vsi_nn_argmin_param* p = &(self->nn_param.argmin);
        p->axis = 2;
        p->keep_dims = FALSE;
    }

    return status;
} /* _argmaxmin_op_init() */


static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* check inputs outputs data type */
    BEGIN_IO_TYPE_DECL(ARGMIN, 1, 1)
        IO_TYPE(D_F16,          D_U8)
        IO_TYPE(D_F16,          D_I16)
        IO_TYPE(D_BF16,         D_U8)
        IO_TYPE(D_BF16,         D_I16)
        IO_TYPE(D_I8|Q_DFP,     D_U8)
        IO_TYPE(D_I8|Q_DFP,     D_I16)
        IO_TYPE(D_U8|Q_ASYM,    D_U8)
        IO_TYPE(D_U8|Q_ASYM,    D_I16)
        IO_TYPE(D_I16|Q_DFP,    D_U8)
        IO_TYPE(D_I16|Q_DFP,    D_I16)
        IO_TYPE(D_F32,          D_I32)
        IO_TYPE(D_F32,          D_I16)
        IO_TYPE(D_F16,          D_I32)
        IO_TYPE(D_I32,          D_I32)
        IO_TYPE(D_I8|Q_DFP,     D_I32)
        IO_TYPE(D_U8|Q_ASYM,    D_I32)
        IO_TYPE(D_I8|Q_ASYM,    D_U8)
        IO_TYPE(D_I8|Q_ASYM,    D_I16)
        IO_TYPE(D_I8|Q_ASYM,    D_I32)
        IO_TYPE(D_I8|Q_SYM,     D_U8)
        IO_TYPE(D_I8|Q_SYM,     D_I16)
        IO_TYPE(D_I8|Q_SYM,     D_I32)
        IO_TYPE(D_I16|Q_ASYM,   D_U8)
        IO_TYPE(D_I16|Q_ASYM,   D_I16)
        IO_TYPE(D_I16|Q_ASYM,   D_I32)
        IO_TYPE(D_I16|Q_SYM,    D_U8)
        IO_TYPE(D_I16|Q_SYM,    D_I16)
        IO_TYPE(D_I16|Q_SYM,    D_I32)
    END_IO_TYPE_DECL(ARGMIN)
    if(!VALIDATE_OP_IO_TYPES(ARGMIN, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_ARG_MAX_MIN_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _argmaxmin_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
            static vsi_bool op_setup_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _argmaxmin_op_setup( ""#kernel_name, self, inputs, outputs ); \
            } \
            static vsi_status op_init_##kernel_name \
                ( \
                vsi_nn_node_t * self \
                ) \
            { \
                return _argmaxmin_op_init( ""#kernel_name, self ); \
            } \
DEF_OP_REG  \
    ( \
    /* op_name    */ name, \
    /* init       */ op_init_##kernel_name, \
    /* compute    */ op_compute_##kernel_name, \
    /* deinit     */ vsi_nn_op_common_deinit, \
    /* check      */ op_check, \
    /* setup      */ op_setup_##kernel_name, \
    /* optimize   */ NULL, \
    /* input_num  */ 1, \
    /* output_num */ 1 \
    )
/*            DEF_OP_REG(name, op_init_##kernel_name, op_compute_##kernel_name, \
                    NULL, NULL, op_setup_##kernel_name, NULL, 1, 1)*/

DEF_ARG_MAX_MIN_OP( ARGMAX, argmax );
DEF_ARG_MAX_MIN_OP( ARGMIN, argmin );


#undef DEF_ARG_MAX_MIN_OP

#ifdef __cplusplus
}
#endif

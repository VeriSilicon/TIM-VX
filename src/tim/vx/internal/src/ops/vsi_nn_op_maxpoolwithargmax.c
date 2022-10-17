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

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (2)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    uint32_t new_rank = 0;
    vsi_nn_kernel_param_t * param = NULL;
    int32_t ksize_x    = (int32_t)self->nn_param.pool.ksize[0];
    int32_t ksize_y    = (int32_t)self->nn_param.pool.ksize[1];
    int32_t stride_x   = (int32_t)self->nn_param.pool.stride[0];
    int32_t stride_y   = (int32_t)self->nn_param.pool.stride[1];
    int32_t pad_left   = (int32_t)self->nn_param.pool.pad[0];
    int32_t pad_right  = (int32_t)self->nn_param.pool.pad[1];
    int32_t pad_top    = (int32_t)self->nn_param.pool.pad[2];
    int32_t pad_bottom = (int32_t)self->nn_param.pool.pad[3];

    if ( NULL == self )
    {
        return VSI_FAILURE;
    }

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_optimize_nchw2xhw_shape(inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shapes[0], &new_rank);
    vsi_nn_kernel_optimize_nchw2xhw_shape(outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[1], &new_rank);
    if (new_rank == 3 && shapes[1][2] == 1)
    {
        new_rank = 2;
    }

    vsi_nn_kernel_param_add_int32( param, "ksize_x",  ksize_x );
    vsi_nn_kernel_param_add_int32( param, "ksize_y",  ksize_y );
    vsi_nn_kernel_param_add_int32( param, "stride_x", stride_x );
    vsi_nn_kernel_param_add_int32( param, "stride_y", stride_y );
    vsi_nn_kernel_param_add_int32( param, "pad_left", pad_left );
    vsi_nn_kernel_param_add_int32( param, "pad_right", pad_right );
    vsi_nn_kernel_param_add_int32( param, "pad_top", pad_top );
    vsi_nn_kernel_param_add_int32( param, "pad_bottom", pad_bottom );

    reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
            inputs[0], shapes[0], new_rank );
    reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], shapes[1], new_rank );
    reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
            outputs[1], shapes[1], new_rank );
    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "maxpoolwithargmax",
            &reshape_tensors[0], _INPUT_NUM, &reshape_tensors[1], _OUTPUT_NUM, param );

    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);
    vsi_safe_release_tensor(reshape_tensors[2]);

    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
    }

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(MAXPOOLWITHARGMAX, 1, 2)
        IO_TYPE(D_I32,   D_I32,  D_I32)
        IO_TYPE(D_F32,   D_F32,  D_I32)
        IO_TYPE(D_F16,   D_F16,  D_I32)
        IO_TYPE(D_BF16, D_BF16,  D_I32)
        IO_TYPE(D_I16|Q_DFP,   D_I16|Q_DFP,  D_I32)
        IO_TYPE(D_U8|Q_ASYM,   D_U8|Q_ASYM,  D_I32)
        IO_TYPE(D_I8|Q_DFP,    D_I8|Q_DFP,   D_I32)
    END_IO_TYPE_DECL(MAXPOOLWITHARGMAX)
    if (!VALIDATE_OP_IO_TYPES(MAXPOOLWITHARGMAX, self, inputs, self->input.num, outputs, self->output.num)) {
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
    vsi_bool ret = TRUE;
    vsi_size_t ksize[_cnt_of_array(self->nn_param.pool.ksize)] = {0};
    vsi_size_t i = 0;
    vsi_size_t pad[_cnt_of_array(self->nn_param.pool.pad)] = {0};
    for (i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
    {
        ksize[i] = self->nn_param.pool.ksize[i];
    }
    for (i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
    {
        pad[i] = self->nn_param.pool.pad[i];
    }

    vsi_nn_compute_padding(
        inputs[0]->attr.size,
        ksize,
        self->nn_param.pool.stride,
        NULL,
        self->nn_param.pool.pad_type,
        pad
    );
    for (i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
    {
        self->nn_param.pool.ksize[i] = (uint32_t)ksize[i];
    }
    for (i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
    {
        self->nn_param.pool.pad[i] = (uint32_t)pad[i];
    }

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        ret = vsi_nn_OpSetup( VSI_NN_OP_POOL, self, inputs, outputs );
    }
    if ( VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num )
    {
        outputs[1]->attr.dim_num = outputs[0]->attr.dim_num;
        memcpy( outputs[1]->attr.size, outputs[0]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t) );
    }
    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MAXPOOLWITHARGMAX,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS


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
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _col2im_local_data_t {
    int32_t placeholder;
} col2im_local_data_t;

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
    vsi_nn_kernel_param_t* param = NULL;
    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "stride_w", self->nn_param.col2im.strides[0] );
    vsi_nn_kernel_param_add_int32( param, "stride_h", self->nn_param.col2im.strides[1] );
    vsi_nn_kernel_param_add_int32( param, "stride_d", self->nn_param.col2im.strides[2] );
    vsi_nn_kernel_param_add_int32( param, "pad_w_front", self->nn_param.col2im.pads[0] );
    vsi_nn_kernel_param_add_int32( param, "pad_w_end", self->nn_param.col2im.pads[1] );
    vsi_nn_kernel_param_add_int32( param, "pad_h_front", self->nn_param.col2im.pads[2] );
    vsi_nn_kernel_param_add_int32( param, "pad_h_end", self->nn_param.col2im.pads[3] );
    vsi_nn_kernel_param_add_int32( param, "pad_d_front", self->nn_param.col2im.pads[4] );
    vsi_nn_kernel_param_add_int32( param, "pad_d_end", self->nn_param.col2im.pads[5] );
    vsi_nn_kernel_param_add_int32( param, "dilation_w", self->nn_param.col2im.dilations[0] );
    vsi_nn_kernel_param_add_int32( param, "dilation_h", self->nn_param.col2im.dilations[1] );
    vsi_nn_kernel_param_add_int32( param, "dilation_d", self->nn_param.col2im.dilations[2] );
    vsi_nn_kernel_param_add_buffer( param, "block_shape", (void*)self->nn_param.col2im.block_shape, \
                                    self->nn_param.col2im.dim_num );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "col2im",
        inputs, 1, outputs, 1, param );

    if (self->n)
    {
        status = VSI_SUCCESS;
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
    BEGIN_IO_TYPE_DECL(COL2IM, 1, 1)
        IO_TYPE(D_F32,        D_F32)
        IO_TYPE(D_F32,        D_I32)
        IO_TYPE(D_F32,        D_U32)
        IO_TYPE(D_F32,        D_F16)
        IO_TYPE(D_I32,        D_F32)
        IO_TYPE(D_I32,        D_I32)
        IO_TYPE(D_I32,        D_U32)
        IO_TYPE(D_I32,        D_F16)
        IO_TYPE(D_U32,        D_F32)
        IO_TYPE(D_U32,        D_I32)
        IO_TYPE(D_U32,        D_U32)
        IO_TYPE(D_F16,        D_I16|Q_DFP)
        IO_TYPE(D_F16,        D_I16|Q_ASYM)
        IO_TYPE(D_F16,        D_I16|Q_SYM)
        IO_TYPE(D_F16,        D_I8|Q_DFP)
        IO_TYPE(D_F16,        D_I8|Q_ASYM)
        IO_TYPE(D_F16,        D_I8|Q_SYM)
        IO_TYPE(D_F16,        D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM, D_F16)
        IO_TYPE(D_I16|Q_ASYM, D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_F16)
        IO_TYPE(D_I16|Q_SYM,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_SYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I16,        D_F16)
        IO_TYPE(D_I16,        D_I8|Q_DFP)
        IO_TYPE(D_I16,        D_U8|Q_ASYM)
        IO_TYPE(D_I16,        D_I32)
        IO_TYPE(D_I16,        D_U32)
        IO_TYPE(D_I16,        D_F32)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_ASYM,  D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_F16)
        IO_TYPE(D_I8|Q_SYM,   D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_SYM,   D_U8|Q_ASYM)
        IO_TYPE(D_I8,         D_F16)
        IO_TYPE(D_I8,         D_I16|Q_DFP)
        IO_TYPE(D_I8,         D_U8|Q_ASYM)
        IO_TYPE(D_I8,         D_I32)
        IO_TYPE(D_I8,         D_U32)
        IO_TYPE(D_I8,         D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_I8|Q_DFP)
        IO_TYPE(D_U8,         D_F16)
        IO_TYPE(D_U8,         D_I16|Q_DFP)
        IO_TYPE(D_U8,         D_I8|Q_DFP)
        IO_TYPE(D_U8,         D_I32)
        IO_TYPE(D_U8,         D_U32)
        IO_TYPE(D_U8,         D_F32)
        IO_TYPE(D_F32,        D_I16|Q_DFP)
        IO_TYPE(D_F32,        D_I16|Q_ASYM)
        IO_TYPE(D_F32,        D_I16|Q_SYM)
        IO_TYPE(D_F32,        D_I8|Q_DFP)
        IO_TYPE(D_F32,        D_I8|Q_ASYM)
        IO_TYPE(D_F32,        D_I8|Q_SYM)
        IO_TYPE(D_F32,        D_U8|Q_ASYM)
        IO_TYPE(D_I32,        D_I16|Q_DFP)
        IO_TYPE(D_I32,        D_I16|Q_ASYM)
        IO_TYPE(D_I32,        D_I16|Q_SYM)
        IO_TYPE(D_I32,        D_I8|Q_DFP)
        IO_TYPE(D_I32,        D_I8|Q_ASYM)
        IO_TYPE(D_I32,        D_I8|Q_SYM)
        IO_TYPE(D_I32,        D_U8|Q_ASYM)
        IO_TYPE(D_F16,        D_F32)
        IO_TYPE(D_F16,        D_I32)
        IO_TYPE(D_F16,        D_I16)
        IO_TYPE(D_F16,        D_U8)
        IO_TYPE(D_F16,        D_I8)
        IO_TYPE(D_F16,        D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,  D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM, D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_I16|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_I32)
        IO_TYPE(D_BF16,       D_BF16)
    END_IO_TYPE_DECL(COL2IM)
    if (!VALIDATE_OP_IO_TYPES(COL2IM, self, inputs, self->input.num, outputs, self->output.num)) {
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
    vsi_nn_col2im_param *p = NULL;
    p = (vsi_nn_col2im_param* )&(self->nn_param.col2im);
    int32_t i = 0;
    vsi_size_t block_size = 1;
    vsi_size_t channel = 1;
    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        outputs[0]->attr.dim_num = p->dim_num + 2;
        for (i = 0; i < p->dim_num; i++)
        {
            outputs[0]->attr.size[i] = (vsi_size_t)p->image_shape[i];
            block_size = block_size * (vsi_size_t)p->block_shape[i];
        }
        channel = inputs[0]->attr.size[1] / block_size;
        outputs[0]->attr.size[i + 1] = channel;
        outputs[0]->attr.size[i + 2] = inputs[0]->attr.size[0];

    }
    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    self->nn_param.col2im.pads[0] = 0;
    self->nn_param.col2im.pads[1] = 0;
    self->nn_param.col2im.pads[2] = 0;
    self->nn_param.col2im.pads[3] = 0;
    self->nn_param.col2im.pads[4] = 0;
    self->nn_param.col2im.pads[5] = 0;
    self->nn_param.col2im.strides[0] = 1;
    self->nn_param.col2im.strides[1] = 1;
    self->nn_param.col2im.strides[2] = 1;
    self->nn_param.col2im.dilations[0] = 1;
    self->nn_param.col2im.dilations[1] = 1;
    self->nn_param.col2im.dilations[2] = 1;

    return VSI_SUCCESS;
}

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ COL2IM,
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


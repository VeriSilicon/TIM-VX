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
#include "utils/vsi_nn_constraint_check.h"
#include "kernel/vsi_nn_kernel.h"

typedef struct _pad2_local_data_t {
    int32_t placeholder;
} pad2_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static int32_t _get_vx_pad_mode(vx_enum mode)
{
    int32_t pad_mode = 0;
    switch (mode)
    {
    case VSI_NN_PAD_MODE_CONSTANT:
        pad_mode = VX_PAD_CONSTANT;
        break;
    case VSI_NN_PAD_MODE_REPLICATE:
        pad_mode = VX_PAD_REPLICATE;
        break;
    case VSI_NN_PAD_MODE_SYMMETRIC:
        pad_mode = VX_PAD_MIRROR_SYMMETRIC;
        break;
    case VSI_NN_PAD_MODE_REFLECT:
        pad_mode = VX_PAD_MIRROR_REFLECT;
        break;
    default:
        VSILOGE("Wrong pad_mode value");
        break;
    }

    return pad_mode;
}

static int32_t _check_mirror_pad_size
    (
    vx_enum mode,
    const uint32_t * front_size,
    const uint32_t * back_size,
    uint32_t pad_dim,
    vsi_size_t *input_size,
    uint32_t tensor_dim
    )
{
    uint32_t dim = pad_dim > tensor_dim ? tensor_dim : pad_dim;
    uint32_t i = 0;

    for (i = 0; i < dim; i++)
    {
        uint32_t front = front_size[i];
        uint32_t end = back_size[i];
        uint32_t sz = (uint32_t)input_size[i];

        if (mode == VSI_NN_PAD_MODE_SYMMETRIC)
        {
            if (front > sz || end > sz)
            {
                VSILOGE("MIRROR SYMMETRIC PAD:each padding value must be less than \
                    or equal to the corresponding dimension");
                return FALSE;
            }
        }
        else if (mode == VSI_NN_PAD_MODE_REFLECT)
        {
            if (front >= sz || end >= sz)
            {
                VSILOGE("MIRROR REFLECT PAD:each padding value must be less than \
                    the corresponding dimension");
                return FALSE;
            }
        }
    }

    return TRUE;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_pad2_param *p = &self->nn_param.pad2;
    vsi_nn_kernel_param_t * param;
    int32_t pad_mode = _get_vx_pad_mode(p->mode);

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_buffer( param, "front_size", (void *)p->front_size, p->dim_num );
    vsi_nn_kernel_param_add_buffer( param, "back_size", (void *)p->back_size, p->dim_num );
    vsi_nn_kernel_param_add_int32( param, "pad_mode", pad_mode );
    vsi_nn_kernel_param_add_float32( param, "const_val", p->const_val );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "pad2",
        inputs, 1, outputs, 1, param );

    if( self->n )
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
    vsi_bool ret = FALSE;
    vsi_nn_pad2_param *p = &self->nn_param.pad2;

    BEGIN_IO_TYPE_DECL(PAD2, 1, 1)
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_F32,          D_BF16)
        IO_TYPE(D_BF16,         D_F32)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_I32,          D_I32)

        /* HW 9.1.1 */
        IO_TYPE(D_U4|Q_ASYM,    D_U4|Q_ASYM)
        IO_TYPE(D_U4|Q_SYM,     D_U4|Q_SYM)
        IO_TYPE(D_I4|Q_ASYM,    D_I4|Q_ASYM)
        IO_TYPE(D_I4|Q_SYM,     D_I4|Q_SYM)

    END_IO_TYPE_DECL(PAD2)
    if (!VALIDATE_OP_IO_TYPES(PAD2, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    if (self->nn_param.pad2.dim_num != inputs[0]->attr.dim_num
        && self->nn_param.pad2.dim_num != 0 )
    {
        VSILOGE("Error:input tensor dim should be equal with pad's.");
        return FALSE;
    }

    ret = _check_mirror_pad_size(p->mode, p->front_size, p->back_size, p->dim_num,
        inputs[0]->attr.size, inputs[0]->attr.dim_num);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i = 0;
    if (self->nn_param.pad2.dim_num == 0)
    {
        self->nn_param.pad2.dim_num = (uint8_t)inputs[0]->attr.dim_num;
    }
    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        for (i = 0; i < self->nn_param.pad2.dim_num; i ++)
        {
            uint32_t front = self->nn_param.pad2.front_size[i];
            uint32_t back  = self->nn_param.pad2.back_size[i];
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i] + front + back;
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    else
    {
        for (i = 0; i < self->nn_param.pad2.dim_num; i ++)
        {
            uint32_t front = self->nn_param.pad2.front_size[i];
            uint32_t back  = self->nn_param.pad2.back_size[i];

            if (front + back + inputs[0]->attr.size[i] != outputs[0]->attr.size[i])
            {
                VSILOGE("Error:output shape[%u] not equal front padding[%u] + input shape[%u] + back padding[%u]",
                    outputs[0]->attr.size[i], front, back);
                return FALSE;
            }
        }
    }
    return TRUE;
} /* op_setup() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PAD2,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

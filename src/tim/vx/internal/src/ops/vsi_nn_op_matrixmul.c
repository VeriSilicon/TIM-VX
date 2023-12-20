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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"


#define _ARG_NUM            (7)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status             status            = VSI_FAILURE;
    vsi_nn_kernel_param_t *param             = NULL;
    vsi_nn_kernel_node_t   n                 = NULL;

    int32_t transposeA  = self->nn_param.matrixmul.transpose[0];
    int32_t transposeB  = self->nn_param.matrixmul.transpose[1];
    int32_t adjointA    = self->nn_param.matrixmul.adjoint[0];
    int32_t adjointB    = self->nn_param.matrixmul.adjoint[1];

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "transposeA", transposeA );
    vsi_nn_kernel_param_add_int32( param, "transposeB", transposeB );
    vsi_nn_kernel_param_add_int32( param, "adjointA", adjointA );
    vsi_nn_kernel_param_add_int32( param, "adjointB", adjointB );

    n = vsi_nn_kernel_selector( self->graph, "matrixmul", inputs, 2, outputs, 1, param );
    if ( n != NULL )
    {
        self->n = (vx_node)n;
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
    vx_bool status = TRUE;

    BEGIN_IO_TYPE_DECL(MATRIXMUL, 2, 1)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_F16,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16,  D_F16)
        IO_TYPE(D_F16,  D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_F16,  D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_F16,  D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_F16,  D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_U8,   D_U8)
        IO_TYPE(D_F16,  D_I8,   D_I8)
        IO_TYPE(D_F16,  D_F16,  D_F16)
        IO_TYPE(D_F32,  D_F32,  D_F32)
        IO_TYPE(D_F32,  D_I8|Q_DFP,  D_F32)
        IO_TYPE(D_F32,  D_I16|Q_DFP,  D_F32)
        IO_TYPE(D_F32,  D_I32,  D_F32)
        IO_TYPE(D_F16,  D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_BF16, D_BF16, D_BF16)
        IO_TYPE(D_I32,  D_I32,  D_I32)
    END_IO_TYPE_DECL(MATRIXMUL)
    if (!VALIDATE_OP_IO_TYPES(MATRIXMUL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    if ((inputs[0]->attr.dim_num == 1 || inputs[1]->attr.dim_num == 1)
        && (self->nn_param.matrixmul.transpose[0] == TRUE || self->nn_param.matrixmul.transpose[1] == TRUE))
    {
         VSILOGE("Transpose parameters should be all false when input tensor is 1D");
         return FALSE;
    }
    else if (self->nn_param.matrixmul.transpose[0] == FALSE
        && self->nn_param.matrixmul.transpose[1] == FALSE
        && inputs[0]->attr.size[0] != inputs[1]->attr.size[1]
        && inputs[0]->attr.dim_num > 1 && inputs[1]->attr.dim_num > 1)
    {
         VSILOGE("1st input tensor's size[0] is not equal to 2nd input tensor's size[1]");
         return FALSE;
    }
    else if (self->nn_param.matrixmul.transpose[0] == TRUE
        && self->nn_param.matrixmul.transpose[1] == FALSE
        && inputs[0]->attr.size[1] != inputs[1]->attr.size[1]
        && inputs[0]->attr.dim_num > 1 && inputs[1]->attr.dim_num > 1)
    {
         VSILOGE("1st input tensor's size[1] is not equal to 2nd input tensor's size[1]");
         return FALSE;
    }
    else if (self->nn_param.matrixmul.transpose[0] == FALSE
        && self->nn_param.matrixmul.transpose[1] == TRUE
        && inputs[0]->attr.size[0] != inputs[1]->attr.size[0]
        && inputs[0]->attr.dim_num > 1 && inputs[1]->attr.dim_num > 1)
    {
         VSILOGE("1st input tensor's size[0] is not equal to 2nd input tensor's size[0]");
         return FALSE;
    }

    if (inputs[0]->attr.dim_num > 2 && inputs[1]->attr.dim_num > 2
        && inputs[0]->attr.size[2] != 1 && inputs[1]->attr.size[2] != 1
        && inputs[0]->attr.size[2] != inputs[1]->attr.size[2])
    {
         VSILOGE("illegal inputs shape");
         return FALSE;
    }

    return status;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i = 0;
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = vsi_nn_max(inputs[0]->attr.dim_num, inputs[1]->attr.dim_num);

        if (node->nn_param.matrixmul.transpose[0] == FALSE
            && node->nn_param.matrixmul.transpose[1] == FALSE)
        {
            outputs[0]->attr.size[0] = inputs[1]->attr.size[0];
            outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        }
        else if (node->nn_param.matrixmul.transpose[0] == TRUE
            && node->nn_param.matrixmul.transpose[1] == FALSE)
        {
            outputs[0]->attr.size[0] = inputs[1]->attr.size[0];
            outputs[0]->attr.size[1] = inputs[0]->attr.size[0];
        }
        else if (node->nn_param.matrixmul.transpose[0] == FALSE
            && node->nn_param.matrixmul.transpose[1] == TRUE)
        {
            outputs[0]->attr.size[0] = inputs[1]->attr.size[1];
            outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        }
        else
        {
            VSILOGE("Not support transpose A and B both TRUE!(MATRIXMUL) at [%s : %d]\n", __FILE__, __LINE__);
            return FALSE;
        }

        if (inputs[0]->attr.dim_num == 1 && inputs[1]->attr.dim_num > 1)
        {
            outputs[0]->attr.dim_num = inputs[1]->attr.dim_num - 1;
            outputs[0]->attr.size[0] = inputs[1]->attr.size[0];
            for (i = 1; i < outputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[1]->attr.size[i + 1];
            }
        }
        else if (inputs[1]->attr.dim_num == 1 && inputs[0]->attr.dim_num > 1)
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num - 1;

            for (i = 0; i < outputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i + 1];
            }
        }
        else
        {
            uint32_t rank0 = inputs[0]->attr.dim_num;
            uint32_t rank1 = inputs[1]->attr.dim_num;
            for (i = 2; i < outputs[0]->attr.dim_num; i++)
            {
                vsi_size_t sz0 = i < rank0 ? inputs[0]->attr.size[i] : 1;
                vsi_size_t sz1 = i < rank1 ? inputs[1]->attr.size[i] : 1;
                vsi_size_t sz2 = vsi_nn_max(sz0, sz1);

                outputs[0]->attr.size[i] = sz2;
            }
        }
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MATRIXMUL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

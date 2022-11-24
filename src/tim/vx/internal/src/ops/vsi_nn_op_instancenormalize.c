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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)

static vsi_status _try_set_high_presision_tensor
    (
    vsi_nn_tensor_t **inputs
    )
{
    vsi_status status;
    vsi_nn_vxtensor_attr_t attr;

    status = VSI_SUCCESS;
    attr = VSI_NN_TENSOR_ATTR_HIGH_PRECISION;

    if(VSI_NN_TYPE_FLOAT32 == inputs[1]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[1], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[2]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[2], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }

    return status;
}

static void vsi_nn_optimize_instance_norm_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    vsi_size_t* out_shape_x, vsi_size_t* out_rank_x
    )
{
    vsi_size_t rank = rank_x;
    vsi_size_t shape[2][VSI_NN_MAX_DIM_NUM] = { {0} };

    if (rank_x > 4)
    {
        memcpy(shape[0], shape_x, (rank_x - 2) * sizeof(vsi_size_t));

        vsi_nn_kernel_optimize_element_shape(shape[0], rank_x - 2, shape[1], &rank);
    }

    if (rank_x == 3)
    {
        out_shape_x[0] = shape_x[0];
        out_shape_x[1] = 1;
        out_shape_x[2] = shape_x[1];
        out_shape_x[3] = shape_x[2];

        *out_rank_x = 4;
    }
    /****reshape [n, c, d0, d1, ..., dn] to [n, c, h, w]***/
    else if (rank_x > 4 && rank == 2)
    {
        memcpy(out_shape_x, shape[1], 2 * sizeof(vsi_size_t));
        memcpy(&out_shape_x[2], &shape_x[rank_x - 2], 2 * sizeof(vsi_size_t));

        *out_rank_x = 4;
    }
    else
    {
        memcpy(out_shape_x, shape_x, rank_x * sizeof(vsi_size_t));

        *out_rank_x = rank_x;
    }
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
    vsi_nn_kernel_node_t    n = NULL;
    float eps = self->nn_param.instancenorm.eps;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_rank = 0;
    vsi_nn_tensor_t * tmp_tensors[4] = {NULL};

    vsi_nn_optimize_instance_norm_shape(inputs[0]->attr.size, inputs[0]->attr.dim_num, shape, &new_rank);

    tmp_tensors[0] = vsi_nn_reshape_tensor( self->graph,
        inputs[0], shape, new_rank );
    tmp_tensors[1] = inputs[1];
    tmp_tensors[2] = inputs[2];
    tmp_tensors[3] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], shape, new_rank );

    status = _try_set_high_presision_tensor(tmp_tensors);
    if(status != VSI_SUCCESS)
    {
        VSILOGE("Set tensor attr of high presision fail");
        return status;
    }

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_float32( param, "eps", eps );

    n = vsi_nn_kernel_selector( self->graph, "instance_norm",
                    tmp_tensors, _INPUT_NUM, &tmp_tensors[3], _OUTPUT_NUM, param );
    if( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
    }

    vsi_safe_release_tensor(tmp_tensors[0]);
    vsi_safe_release_tensor(tmp_tensors[3]);

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(INSTANCE_NORM, 3, 1)
        IO_TYPE(D_F16,        D_F32,  D_F16,  D_F16)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F16)
        IO_TYPE(D_F16,        D_F16,  D_F16,  D_F16)
        IO_TYPE(D_F32,        D_F32,  D_F16,  D_F32)
        IO_TYPE(D_F32,        D_F16,  D_F16,  D_F32)
        IO_TYPE(D_F32,        D_F32,  D_F32,  D_F32)
        IO_TYPE(D_I32,        D_F32,  D_F16,  D_I32)
        IO_TYPE(D_I32,        D_F32,  D_F16,  D_F32)
        IO_TYPE(D_BF16,       D_F32,  D_F32,  D_BF16)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F16,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F32,  D_F16)
        IO_TYPE(D_I8|Q_ASYM,  D_F32,  D_F32,  D_F16)
        IO_TYPE(D_I8|Q_SYM,   D_F32,  D_F32,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F32,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,  D_F32,  D_F32,  D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_F32,  D_F32,  D_I8|Q_SYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_I8|Q_DFP)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_I8|Q_ASYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_I8|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F32,  D_F16)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F32,  D_F16)
        IO_TYPE(D_I16|Q_ASYM, D_F32,  D_F32,  D_F16)
        IO_TYPE(D_I16|Q_SYM,  D_F32,  D_F32,  D_F16)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_I16|Q_DFP)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_I16|Q_ASYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F32,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM, D_F32,  D_F32,  D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_F32,  D_F32,  D_I16|Q_SYM)
    END_IO_TYPE_DECL(INSTANCE_NORM)
    if (!VALIDATE_OP_IO_TYPES(INSTANCE_NORM, self, inputs, self->input.num, outputs, self->output.num))
    {
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
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ INSTANCE_NORM,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

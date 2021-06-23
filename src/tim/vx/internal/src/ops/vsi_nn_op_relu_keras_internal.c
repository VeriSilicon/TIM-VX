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
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

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
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    int32_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    int32_t new_rank = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_relu_keras_internal_param * p = NULL;
    float     alpha     = 0.0f;
    float     max_value = 0.0f;
    float     threshold = 0.0f;

    if( NULL == self )
    {
        return status;
    }

    p          = &(self->nn_param.relu_keras_internal);
    alpha      = p->alpha;
    max_value  = p->max_value;
    threshold  = p->threshold;
    param      = vsi_nn_kernel_param_create();

    ret = vsi_nn_kernel_optimize_element_shape(
            (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );

    vsi_nn_kernel_param_add_float32( param, "alpha",  alpha );
    vsi_nn_kernel_param_add_float32( param, "max_value",  max_value );
    vsi_nn_kernel_param_add_float32( param, "threshold",  threshold );

    if( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], (uint32_t*)shape, new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], (uint32_t*)shape, new_rank );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "relu_keras",
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

} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(RELU_KERAS_INTERNAL, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
    END_IO_TYPE_DECL(RELU_KERAS_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(RELU_KERAS_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_RELU_KERAS_INTERNAL_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.relu_keras_internal.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.relu_keras_internal.local.local_tensor[i]));
            self->nn_param.relu_keras_internal.local.local_tensor[i] = NULL;
        }
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RELU_KERAS_INTERNAL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

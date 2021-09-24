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
#include "vsi_nn_error.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define VSI_NN_PRELU_DEFAULT_AXIS 2

static vsi_bool _is_one_rank_tensor
    (
    vsi_nn_tensor_t * input,
    vsi_size_t *shape
    )
{
    uint32_t i = 0;
    uint32_t one_rank = 0;

    *shape = 1;

    for (i = 0; i < input->attr.dim_num; i++)
    {
        if (input->attr.size[i] != 1)
        {
            *shape = input->attr.size[i];
            one_rank ++;
        }
    }

    if (one_rank <= 1)
    {
        return TRUE;
    }

    return FALSE;
}

static vsi_status _prelu_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_prelu_param *prelu = &self->nn_param.prelu;
    vsi_ssize_t shapes[VSI_NN_MAX_DIM_NUM] = { 1 };
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_bool   one_rank = FALSE;
    vsi_bool   is_per_channel_alpha = 0;
    vsi_size_t alpha_shape = 1;
    uint32_t i = 0;
    vsi_nn_kernel_param_t * param = NULL;
    uint32_t dims = outputs[0]->attr.dim_num;

    reshape_tensors[0] = inputs[0];
    one_rank = _is_one_rank_tensor(inputs[1], &alpha_shape);

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        shapes[i] = 1;
    }

    if (vsi_nn_compareVersion(self->graph, 1, 1, 20) == -1)
    {
        int32_t axis = prelu->axis;

        if (one_rank)
        {
            is_per_channel_alpha = (alpha_shape == 1) || axis == 2;

            if (is_per_channel_alpha)
            {
                shapes[0] = alpha_shape;
                dims = 2;
            }
            else
            {
               memcpy(shapes, inputs[1]->attr.size, inputs[1]->attr.dim_num * sizeof(vsi_size_t));
               dims = inputs[1]->attr.dim_num;
            }

            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], (vsi_size_t*)shapes, dims );
        }
        else
        {
            memcpy(shapes, inputs[1]->attr.size, inputs[1]->attr.dim_num * sizeof(vsi_size_t));
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], (vsi_size_t*)shapes, inputs[1]->attr.dim_num );
        }
    }
    else
    {
        dims = inputs[1]->attr.dim_num;

        memcpy(shapes, inputs[1]->attr.size, inputs[1]->attr.dim_num * sizeof(vsi_size_t));

        if (one_rank)
        {
            is_per_channel_alpha = (inputs[1]->attr.dim_num > 2 && alpha_shape == inputs[1]->attr.size[2]);
        }

        if (is_per_channel_alpha)
        {
            shapes[0] = alpha_shape;
            shapes[1] = 1;
            dims = 2;
        }

        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
            inputs[1], (vsi_size_t*)shapes, dims );
    }

    // Add params
    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "is_per_channel_alpha", is_per_channel_alpha );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        kernel_name,
        &reshape_tensors[0], 2,
        outputs, 1, param );

    vsi_nn_kernel_param_release( &param );
    vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
} /* _prelu_op_compute() */

vsi_bool vsi_nn_op_prelu_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_prelu_param *prelu = &self->nn_param.prelu;

    if( NULL == self )
    {
        return FALSE;
    }

    if (prelu->axis < 0)
    {
        prelu->axis += (int32_t)inputs[0]->attr.dim_num;
    }

    if (prelu->axis < 0)
    {
        VSILOGD("PRelu Invalid Axis: %d \n", prelu->axis);
        return FALSE;
    }

    if (vsi_nn_compareVersion(self->graph, 1, 1, 20) == -1)
    {
        ret = vsi_nn_op_common_setup(self, inputs, outputs);
    }
    else
    {
        ret = vsi_nn_OpSetup( VSI_NN_OP_MULTIPLY, self, inputs, outputs );
    }

    return ret;
} /* vsi_nn_op_prelu_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(PRELU, 2, 1)
        IO_TYPE(D_F16,  D_F16, D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_F16, D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_F16, D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_F16, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_F16, D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_F16, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_F16, D_F16)
        IO_TYPE(D_I16|Q_DFP, D_F16, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP, D_F16, D_F16)
        IO_TYPE(D_BF16, D_F16, D_BF16)
        IO_TYPE(D_BF16, D_BF16, D_BF16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_F32, D_F32, D_F32)
        IO_TYPE(D_I32, D_I32, D_I32)

        /* HW 9.0 */
        IO_TYPE(D_F32, D_BF16, D_BF16)
        IO_TYPE(D_BF16, D_BF16, D_F32)
    END_IO_TYPE_DECL(PRELU)
    if(!VALIDATE_OP_IO_TYPES(PRELU, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 17) == -1)
    {
        self->nn_param.prelu.axis = VSI_NN_PRELU_DEFAULT_AXIS;
    }

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_ELEMENT_WISE_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _prelu_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
DEF_OP_REG(name, op_init, op_compute_##kernel_name, vsi_nn_op_common_deinit, \
        op_check, vsi_nn_op_prelu_setup, NULL, 2, 1)

DEF_ELEMENT_WISE_OP( PRELU, prelu );

#undef DEF_ELEMENT_WISE_OP

#ifdef __cplusplus
}
#endif

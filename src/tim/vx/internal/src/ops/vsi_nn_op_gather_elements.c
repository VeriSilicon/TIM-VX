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

typedef struct _gather_elements_local_data_t {
    int32_t placeholder;
} gather_elements_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_nn_tensor_t* temp_tensors = NULL;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    int32_t axis = 0;
    int32_t new_axis0 = 0;
    int32_t new_axis1 = 0;
    vsi_bool ret = FALSE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_gather_elements_param * p = NULL;
    vsi_size_t depth0 = inputs[0]->attr.dim_num > 2 ? inputs[0]->attr.size[2] : 1;
    vsi_size_t depth1 = inputs[1]->attr.dim_num > 2 ? inputs[1]->attr.size[2] : 1;

    if ( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;

    p = &(self->nn_param.gather_elements);
    axis = p->axis;

    ret = vsi_nn_kernel_optimize_softmax_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num, axis,
            shapes[0], &rank_in, &new_axis0);
    ret |= vsi_nn_kernel_optimize_softmax_shape(
            inputs[1]->attr.size, inputs[1]->attr.dim_num, axis,
            shapes[1], &rank_in, &new_axis1);

    // Add params
    param = vsi_nn_kernel_param_create();

    if (vsi_nn_is_same_type(inputs[0], outputs[0]) == FALSE)
    {
        vsi_nn_tensor_attr_t attr;

        VSILOGW("gather_element is no_range_change operation! \
            Insert DataConvert Operation when the quantization parameters of input and output are inconsistent!");

        memcpy( &attr, &outputs[0]->attr, sizeof(attr));
        memcpy( &attr.dtype, &inputs[0]->attr.dtype, sizeof(attr.dtype));
        attr.is_const = FALSE;
        attr.vtl = TRUE;
        temp_tensors = vsi_nn_CreateTensor( self->graph, &attr );
    }
    else
    {
        temp_tensors = outputs[0];
    }

    if ( ret && new_axis0 == new_axis1 &&
        inputs[0]->attr.size[0] < GPU_TENSOR_MAX_WIDTH &&
        inputs[0]->attr.size[1] < GPU_TENSOR_MAX_WIDTH &&
        inputs[1]->attr.size[0] < GPU_TENSOR_MAX_WIDTH &&
        inputs[1]->attr.size[1] < GPU_TENSOR_MAX_WIDTH &&
        depth0 < GPU_TENSOR_MAX_WIDTH &&
        depth1 < GPU_TENSOR_MAX_WIDTH)
    {
        vsi_nn_kernel_param_add_int32( param, "axis", new_axis0 );

        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], shapes[1], rank_in );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                temp_tensors, shapes[1], rank_in );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "gather_elements",
                &reshape_tensors[0], 2,
                &reshape_tensors[2], 1, param );

        vsi_safe_release_tensor( reshape_tensors[0] );
        vsi_safe_release_tensor( reshape_tensors[1] );
        vsi_safe_release_tensor( reshape_tensors[2] );
    }
    else
    {
        vsi_nn_kernel_param_add_int32( param, "axis", axis );
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "gather_elements",
                inputs, 2,
                &temp_tensors, 1, param );
    }

    if (vsi_nn_is_same_type(inputs[0], outputs[0]) == FALSE)
    {
        self->n = vxTensorCopyNode( self->graph->g, temp_tensors->t, outputs[0]->t);
        vsi_safe_release_tensor(temp_tensors);
    }

    vsi_nn_kernel_param_release( &param );

    if ( self->n )
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
    BEGIN_IO_TYPE_DECL(GATHER_ELEMENTS, 2, 1)
        IO_TYPE(D_I32,        D_I32, D_I32)
        IO_TYPE(D_F32,        D_I32, D_F32)
        IO_TYPE(D_F16,        D_I32, D_F16)
        IO_TYPE(D_BF16,       D_I32, D_BF16)
        IO_TYPE(D_U8|Q_ASYM,  D_I32, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_I32, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,  D_I32, D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_I32, D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,  D_I32, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM, D_I32, D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_I32, D_I16|Q_SYM)
    END_IO_TYPE_DECL(GATHER_ELEMENTS)
    if (!VALIDATE_OP_IO_TYPES(GATHER_ELEMENTS, self, inputs, self->input.num, outputs, self->output.num))
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
    VSI_UNREFERENCED(self);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t i = 0;
        outputs[0]->attr.dim_num = inputs[1]->attr.dim_num;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
        }
    }

    return TRUE;
} /* op_setup() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GATHER_ELEMENTS,
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

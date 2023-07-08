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

/*
 Declare number of input and output.
 */

static vsi_bool _is_supported_axis(vsi_size_t* multiples, vsi_size_t multiples_num)
{
    vsi_size_t i = 0;

    if ( multiples_num < 4)
    {
        return TRUE;
    }
    else if ( multiples_num > 4)
    {
        return FALSE;
    }

    for ( i = 3;  i < multiples_num;  i++)
    {
        if (multiples[i] > 1)
        {
            return FALSE;
        }
    }

    return TRUE;
}

static vsi_status _tile_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status                        = VSI_FAILURE;
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_size_t new_rank                      = 0;
    vsi_bool   ret                          = FALSE;
    vsi_size_t* multiples                   = (vsi_size_t*)self->nn_param.tile.multiples;
    vsi_nn_tensor_t* temp_tensors[2]        = { NULL };
    vsi_nn_tensor_t* reshape_tensors[2]     = { NULL };
    vsi_nn_tensor_attr_t attr;

    if (vsi_nn_is_same_type(inputs[0], outputs[0]) == FALSE)
    {
        VSILOGW("tile is no_range_change operation! \
            Insert DataConvert Operation when the quantization parameters\
            of input and output are inconsistent!");

        memcpy( &attr, &outputs[0]->attr, sizeof(attr));
        memcpy( &attr.dtype, &inputs[0]->attr.dtype, sizeof(attr.dtype));
        attr.is_const = FALSE;
        attr.vtl = TRUE;
        temp_tensors[1] = vsi_nn_CreateTensor( self->graph, &attr );
    }
    else
    {
        temp_tensors[1] = outputs[0];
    }

    ret = vsi_nn_kernel_optimize_tile_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            multiples, inputs[0]->attr.dim_num,
            temp_tensors[1]->attr.size, temp_tensors[1]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    if (ret)
    {
        if (_is_supported_axis(shapes[1], new_rank) == FALSE)
        {
            reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph, inputs[0],\
                shapes[0], (vsi_size_t)new_rank );
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph, temp_tensors[1],\
                shapes[2], (vsi_size_t)new_rank );
            if (reshape_tensors[0] == NULL || reshape_tensors[1] == NULL)
            {
                VSILOGE("reshape tensor failed!");
                status = VSI_FAILURE;
                goto final;
            }

            memcpy( &attr, &reshape_tensors[0]->attr, sizeof(attr));
            attr.is_const = FALSE;
            attr.vtl = TRUE;
            attr.size[0] = reshape_tensors[1]->attr.size[0];
            attr.size[1] = reshape_tensors[1]->attr.size[1];

            temp_tensors[0] = vsi_nn_CreateTensor( self->graph, &attr );

            self->n = (vx_node)vsi_nn_kernel_selector(
                self->graph, kernel_name, &reshape_tensors[0], 1, &temp_tensors[0], 1, NULL);
            self->n = (vx_node)vsi_nn_kernel_selector(
                self->graph, kernel_name, &temp_tensors[0], 1, &reshape_tensors[1], 1, NULL);

        }
        else
        {
            reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph, inputs[0],\
                shapes[0], (vsi_size_t)new_rank );
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph, temp_tensors[1],\
                shapes[2], (vsi_size_t)new_rank );
            if (reshape_tensors[0] == NULL || reshape_tensors[1] == NULL)
            {
                VSILOGE("reshape tensor failed!");
                status = VSI_FAILURE;
                goto final;
            }

            self->n = (vx_node)vsi_nn_kernel_selector( self->graph, kernel_name,\
                &reshape_tensors[0], 1, &reshape_tensors[1], 1, NULL );
        }
    }

    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

final:
    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);
    vsi_safe_release_tensor(temp_tensors[0]);
    if (vsi_nn_is_same_type(inputs[0], outputs[0]) == FALSE)
    {
        self->n = vxTensorCopyNode( self->graph->g, temp_tensors[1]->t, outputs[0]->t);
        vsi_safe_release_tensor(temp_tensors[1]);
    }

    return status;
} /* _tile_op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes.  */
    vsi_nn_tile_param * p;

    BEGIN_IO_TYPE_DECL(TILE, 1, 1)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_I32,          D_I32)
        IO_TYPE(D_U32,          D_U32)
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_F32,          D_U8|Q_ASYM)
        IO_TYPE(D_F16,          D_U8|Q_ASYM)
    END_IO_TYPE_DECL(TILE)
    if (!VALIDATE_OP_IO_TYPES(TILE, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    p = &(self->nn_param.tile);

    if (inputs[0]->attr.dim_num != p->multiples_num)
    {
        VSILOGE("multiples_num MUST match the dims of input tensor!");
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
    uint32_t i;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        vsi_nn_tile_param * p;

        p = &(self->nn_param.tile);
        if (inputs[0]->attr.dim_num != p->multiples_num)
        {
            VSILOGE("multiples_num MUST match the dims of input tensor!");
            return FALSE;
        }

        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i] * p->multiples[i];
        }
    }

    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_TILE_OP(name, kernel_name) \
    static vsi_status op_compute_##kernel_name \
        ( \
        vsi_nn_node_t * self, \
        vsi_nn_tensor_t ** inputs, \
        vsi_nn_tensor_t ** outputs \
        ) \
    { \
        return _tile_op_compute( ""#kernel_name, self, inputs, outputs ); \
    } \
DEF_OP_REG(name, NULL, op_compute_##kernel_name, vsi_nn_op_common_deinit, op_check, op_setup, NULL, 1, 1)

DEF_TILE_OP( TILE, tile );

#undef DEF_TILE_OP

#ifdef __cplusplus
}
#endif

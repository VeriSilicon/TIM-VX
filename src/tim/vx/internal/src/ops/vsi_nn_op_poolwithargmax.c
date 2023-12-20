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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (2)

static vsi_bool vsi_nn_poolwithargmax_optimize_shape
    (
    vsi_nn_node_t * self,
    const vsi_ssize_t* shape_in, const vsi_ssize_t* shape_out0,
    const vsi_ssize_t* shape_out1, const size_t rank_in,
    vsi_ssize_t* out_shape_input, vsi_ssize_t* out_shape_output0,
    vsi_ssize_t* out_shape_output1, uint32_t* out_rank_output
    )
{
    vsi_bool   enable_image_2d = FALSE;
    int32_t    hwLitimLen      = GPU_TENSOR_MAX_WIDTH;

    if ((2 == self->nn_param.pool.ksize[1])
       && (2 == self->nn_param.pool.stride[1])
       && ((shape_in[1] % 2 == 0) || (shape_in[2] == 1)))
    {
        if (rank_in < 3)
        {
            enable_image_2d = TRUE;
        }
        else
        {
            enable_image_2d = (vsi_bool)(shape_in[1] * shape_in[2] < hwLitimLen);
        }
    }

    if( rank_in == 1 )
    {
        out_shape_input[0]   = shape_in[0];
        out_shape_input[1]   = 1;
        out_shape_input[2]   = 1;
        out_shape_output0[0] = shape_out0[0];
        out_shape_output0[1] = 1;
        out_shape_output0[2] = 1;
        out_shape_output1[0] = shape_out1[0];
        out_shape_output1[1] = 1;
        out_shape_output1[2] = 1;
        *out_rank_output     = 2;
    }
    else if(rank_in == 3 && enable_image_2d)
    {
        out_shape_input[0]   = shape_in[0];
        out_shape_input[1]   = shape_in[1] * shape_in[2];
        out_shape_input[2]   = 1;
        out_shape_output0[0] = shape_out0[0];
        out_shape_output0[1] = shape_out0[1] * shape_out0[2];
        out_shape_output0[2] = 1;
        out_shape_output1[0] = shape_out1[0];
        out_shape_output1[1] = shape_out1[1] * shape_out1[2];
        out_shape_output1[2] = 1;
        *out_rank_output     = 2;
    }
    else if(rank_in == 4 && enable_image_2d)
    {
        out_shape_input[0]   = shape_in[0];
        out_shape_input[1]   = shape_in[1] * shape_in[2];
        out_shape_input[2]   = 1;
        out_shape_input[3]   = shape_in[3];
        out_shape_output0[0] = shape_out0[0];
        out_shape_output0[1] = shape_out0[1] * shape_out0[2];
        out_shape_output0[2] = 1;
        out_shape_output0[3] = shape_out0[3];
        out_shape_output1[0] = shape_out1[0];
        out_shape_output1[1] = shape_out1[1] * shape_out1[2];
        out_shape_output1[2] = 1;
        out_shape_output1[3] = shape_out1[3];
        *out_rank_output     = 4;
    }
    else
    {
        uint32_t i;
        for (i = 0; i < rank_in; i++)
        {
            out_shape_input[i]   = shape_in[i];
            out_shape_output0[i] = shape_out0[i];
            out_shape_output1[i] = shape_out1[i];
        }
        *out_rank_output = (uint32_t)rank_in;
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
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    uint32_t new_rank = 0;
    vsi_bool ret = FALSE;
    vsi_nn_kernel_param_t * param = NULL;
    int32_t ksize_x  = 0;
    int32_t ksize_y  = 0;
    int32_t stride_x = 0;
    int32_t stride_y = 0;
    int32_t pad_x    = 0;
    int32_t pad_y    = 0;

    if ( NULL == self )
    {
        return VSI_FAILURE;
    }

    ksize_x  = (int32_t)self->nn_param.pool.ksize[0];
    ksize_y  = (int32_t)self->nn_param.pool.ksize[1];
    stride_x = (int32_t)self->nn_param.pool.stride[0];
    stride_y = (int32_t)self->nn_param.pool.stride[1];
    pad_x    = (int32_t)self->nn_param.pool.pad[0];
    pad_y    = (int32_t)self->nn_param.pool.pad[2];

    param = vsi_nn_kernel_param_create();

    ret = vsi_nn_poolwithargmax_optimize_shape(self,
            (vsi_ssize_t*)inputs[0]->attr.size,  (vsi_ssize_t*)outputs[0]->attr.size,
            (vsi_ssize_t*)outputs[1]->attr.size, inputs[0]->attr.dim_num,
            (vsi_ssize_t*)shapes[0], (vsi_ssize_t*)shapes[1], (vsi_ssize_t*)shapes[2], &new_rank );

    vsi_nn_kernel_param_add_int32( param, "ksize_x",  ksize_x );
    vsi_nn_kernel_param_add_int32( param, "ksize_y",  ksize_y );
    vsi_nn_kernel_param_add_int32( param, "stride_x", stride_x );
    vsi_nn_kernel_param_add_int32( param, "stride_y", stride_y );
    vsi_nn_kernel_param_add_int32( param, "pad_x",    pad_x );
    vsi_nn_kernel_param_add_int32( param, "pad_y",    pad_y );

    if ( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[1], new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[1], shapes[2], new_rank );
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "poolwithargmax",
                                                 &reshape_tensors[0], _INPUT_NUM,
                                                 &reshape_tensors[1], _OUTPUT_NUM, param );
        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        vsi_nn_ReleaseTensor( &reshape_tensors[2] );
    }

    if ( self->n )
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
    BEGIN_IO_TYPE_DECL(POOLWITHARGMAX, 1, 2)
        IO_TYPE(D_F16,   D_F16,  D_U8)
        IO_TYPE(D_F16,   D_I16|Q_DFP,  D_U8)
        IO_TYPE(D_U8|Q_ASYM,   D_U8|Q_ASYM,  D_U8)
        IO_TYPE(D_U8|Q_ASYM,   D_F16,  D_U8)
        IO_TYPE(D_U8|Q_ASYM,   D_F16,  D_I16)
        IO_TYPE(D_I8|Q_DFP,    D_I8|Q_DFP,  D_U8)
        IO_TYPE(D_I8|Q_DFP,    D_F16,  D_U8)
        IO_TYPE(D_I16|Q_DFP,   D_I16|Q_DFP,  D_U8)
        IO_TYPE(D_I16|Q_DFP,   D_F16,  D_U8)
        IO_TYPE(D_I16|Q_DFP,   D_I16|Q_DFP,  D_I16)
        IO_TYPE(D_F32,   D_F32,  D_U8)
        IO_TYPE(D_F32,   D_U8|Q_ASYM,  D_U8)
        IO_TYPE(D_U8|Q_ASYM,   D_F32,  D_U8)
        IO_TYPE(D_I32,   D_I32,  D_U8)

        /* HW 9.0 */
        IO_TYPE(D_BF16, D_BF16, D_U8)
    END_IO_TYPE_DECL(POOLWITHARGMAX)
    if(!VALIDATE_OP_IO_TYPES(POOLWITHARGMAX, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    if (VX_CONVOLUTIONAL_NETWORK_POOLING_MAX != self->nn_param.pool.type)
    {
        VSILOGE("Unsupported pool type.\n");
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
    vsi_bool ret = TRUE;
    vsi_size_t ksize[_cnt_of_array(self->nn_param.pool.ksize)] = {0};
    vsi_size_t i = 0;
    vsi_size_t pad[_cnt_of_array(self->nn_param.pool.pad)] = {0};
    for(i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
    {
        ksize[i] = self->nn_param.pool.ksize[i];
    }
    for(i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
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
    for(i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
    {
        self->nn_param.pool.ksize[i] = (uint32_t)ksize[i];
    }
    for(i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
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
    vsi_nn_node_t * self
    )
{
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ POOLWITHARGMAX,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

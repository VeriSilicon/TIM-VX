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

#include "vsi_nn_types.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _conv1d_local_data_t {
    vsi_bool use_ext_pad;
    vsi_bool use_ovxlib_kernel;
    vsi_nn_internal_tensor_t* pad_output;
} conv1d_local_data_t;

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;

    param = vsi_nn_kernel_param_create();

    if(self->nn_param.conv1d.local->use_ovxlib_kernel)
    {
        vsi_nn_tensor_t* new_inputs[3] = { NULL };
        vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
        uint32_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
        int32_t new_rank = 0;
        int32_t pad_front = self->nn_param.conv1d.pad[0];
        int32_t pad_end   = self->nn_param.conv1d.pad[1];

        if (1 == inputs[0]->attr.dim_num)
        {
            shape[0] = inputs[0]->attr.size[0];
            shape[1] = 1;
            new_rank = 2;
            reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                    inputs[0], (uint32_t*)shape, new_rank );
            new_inputs[0] = reshape_tensors[0];
        }
        else
        {
            new_inputs[0] = inputs[0];
        }

        if (1 == inputs[1]->attr.dim_num)
        {
            shape[0] = inputs[1]->attr.size[0];
            shape[1] = 1;
            new_rank = 2;
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                    inputs[1], (uint32_t*)shape, new_rank );
            new_inputs[1] = reshape_tensors[1];
        }
        else
        {
            new_inputs[1] = inputs[1];
        }

        if (1 == inputs[2]->attr.dim_num)
        {
            shape[0] = inputs[2]->attr.size[0];
            shape[1] = 1;
            new_rank = 2;
            reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                    inputs[2], (uint32_t*)shape, new_rank );
            new_inputs[2] = reshape_tensors[2];
        }
        else
        {
            new_inputs[2] = inputs[2];
        }

        /* overwrite input[0] with padded tensor */
        if(self->nn_param.conv1d.local->use_ext_pad)
        {
            vsi_nn_internal_compute_node( self );
            new_inputs[0] = self->nn_param.conv1d.local->pad_output->t;
            pad_front = 0;
            pad_end   = 0;
        }

        vsi_nn_kernel_param_add_int32( param, "stride", self->nn_param.conv1d.stride );
        vsi_nn_kernel_param_add_int32( param, "pad_front", pad_front );
        vsi_nn_kernel_param_add_int32( param, "pad_end", pad_end );
        vsi_nn_kernel_param_add_int32( param, "dilation", self->nn_param.conv1d.dilation );
        vsi_nn_kernel_param_add_int32( param, "overflow_policy", self->vx_param.overflow_policy );
        vsi_nn_kernel_param_add_int32( param, "rounding_policy", self->vx_param.rounding_policy );
        vsi_nn_kernel_param_add_int32( param,
                "down_scale_size_rounding", self->vx_param.down_scale_size_rounding );
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "conv1d_ovxlib",
                new_inputs, 3, outputs, 1, param );

        if (reshape_tensors[0]) vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        if (reshape_tensors[1]) vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        if (reshape_tensors[2]) vsi_nn_ReleaseTensor( &reshape_tensors[2] );
    }
    else
    {
        vsi_nn_kernel_param_add_int32( param, "stride", self->nn_param.conv1d.stride );
        vsi_nn_kernel_param_add_int32( param, "pad_front", self->nn_param.conv1d.pad[0] );
        vsi_nn_kernel_param_add_int32( param, "pad_end", self->nn_param.conv1d.pad[1] );
        vsi_nn_kernel_param_add_int32( param, "dilation", self->nn_param.conv1d.dilation );
        vsi_nn_kernel_param_add_int32( param, "overflow_policy", self->vx_param.overflow_policy );
        vsi_nn_kernel_param_add_int32( param, "rounding_policy", self->vx_param.rounding_policy );
        vsi_nn_kernel_param_add_int32( param,
                "down_scale_size_rounding", self->vx_param.down_scale_size_rounding );
        if( self->nn_param.conv1d.multiplier > 0 )
        {
            vsi_nn_kernel_param_add_int32( param, "multiplier",
                    self->nn_param.conv1d.multiplier );
            self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "depthwise_conv1d",
                inputs, 3, outputs, 1, param );
        }
        else
        {
            self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "conv1d",
                inputs, 3, outputs, 1, param );
        }
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
    vsi_bool ret = FALSE;

    BEGIN_IO_TYPE_DECL(CONV1D, 3, 1)
        IO_TYPE(D_F16,  D_F16,  D_NONE, D_F16)
        IO_TYPE(D_F16,  D_F16,  D_F32, D_F16)
        IO_TYPE(D_F16,  D_F16,  D_F16, D_F16)
        IO_TYPE(D_F32,  D_F32,  D_F32, D_F32)
        IO_TYPE(D_F32,  D_F32,  D_F32, D_BF16)
        IO_TYPE(D_F32,  D_F32,  D_NONE, D_F32)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_NONE, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I32|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I64|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_NONE, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,  D_I32|Q_DFP, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_NONE, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_BF16,  D_BF16,  D_F32, D_BF16)
        IO_TYPE(D_BF16,  D_BF16,  D_F32, D_F32)
        IO_TYPE(D_BF16,  D_BF16,  D_NONE, D_BF16)
    END_IO_TYPE_DECL(CONV1D)
    if (!VALIDATE_OP_IO_TYPES(CONV1D, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    /* Check fl and scale*/
    ret = vsi_nn_QuantCheck(inputs[0], inputs[1], inputs[2]);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_conv1d_param* p = &self->nn_param.conv1d;

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            inputs[1]->attr.size[0],
            p->pad,
            p->stride,
            p->dilation,
            VSI_NN_ROUND_FLOOR
            );

        outputs[0]->attr.size[1] = inputs[1]->attr.size[2];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.dim_num = 3;
    }

    if ( (self->nn_param.conv1d.ksize == 1024 && self->nn_param.conv1d.dilation == 1)
      || (self->nn_param.conv1d.ksize == 3 && self->nn_param.conv1d.dilation > 7) )
    {
        if (self->nn_param.conv1d.stride == 1 && self->nn_param.conv1d.multiplier == 0)
        {
            self->nn_param.conv1d.local->use_ovxlib_kernel = TRUE;
            if ((p->pad[0] || p->pad[1]) && (inputs[0]->attr.size[0] >= 65535))
            {
                vsi_nn_tensor_attr_t attr;
                vsi_nn_internal_node_t* curr = NULL;
                vsi_nn_internal_tensor_t* tensor = NULL;
                uint32_t *front_data = NULL;
                uint32_t *back_data  = NULL;
                memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
                vsi_nn_internal_init_tensor_attr(&attr, &inputs[0]->attr.dtype, TRUE);
                tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

                curr = vsi_nn_internal_new_node(self, VSI_NN_OP_PAD, 0, 0);
                front_data = (uint32_t*)\
                    vsi_nn_internal_new_node_param(curr, sizeof(uint32_t) * inputs[0]->attr.dim_num);
                back_data = (uint32_t*)\
                    vsi_nn_internal_new_node_param(curr, sizeof(uint32_t) * inputs[0]->attr.dim_num);

                front_data[0] = p->pad[0];
                front_data[1] = 0;
                front_data[2] = 0;
                back_data[0]  = p->pad[1];
                back_data[1]  = 0;
                back_data[2]  = 0;
                curr->node->nn_param.pad.front_size    = front_data;
                curr->node->nn_param.pad.back_size     = back_data;
                curr->node->nn_param.pad.dim_num       = 3;
                curr->node->nn_param.pad.const_val     = 0;
                curr->node->nn_param.pad.mode          = VSI_NN_PAD_MODE_CONSTANT;
                curr->inputs[0]                        = inputs[0];
                curr->outputs[0]                       = tensor->t;
                vsi_nn_internal_setup_node(self, curr);

                self->nn_param.conv1d.local->use_ext_pad = TRUE;
                self->nn_param.conv1d.local->pad_output = tensor;
            }
        }
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_internal_deinit_node_wksp(self);

    vsi_nn_safe_free(self->nn_param.gru_ovxlib.local);

    return vsi_nn_op_common_deinit(self);
}

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_init_node_wksp(self);

    self->nn_param.conv1d.local = (conv1d_local_data_t *)malloc(sizeof(conv1d_local_data_t));
    memset(self->nn_param.conv1d.local, 0x00, sizeof(conv1d_local_data_t));
    self->nn_param.conv1d.local->use_ext_pad = FALSE;
    self->nn_param.conv1d.local->use_ovxlib_kernel = FALSE;

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONV1D,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif


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
#include "utils/vsi_nn_util.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_internal_node.h"
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
    float min_value = self->nn_param.clip.min;
    float max_value = self->nn_param.clip.max;

    if ( (min_value == -1.0f && max_value == 1.0f)
      || (min_value == 0.0f && max_value == 6.0f) )
    {
        status = VSI_SUCCESS;
        vsi_nn_internal_compute_node( self );
    }
    else
    {
        vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
        vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
        vsi_size_t new_rank = 0;
        vsi_bool ret;
        vsi_nn_kernel_param_t * param = NULL;

        param =vsi_nn_kernel_param_create();

        ret = vsi_nn_kernel_optimize_element_shape(
                inputs[0]->attr.size, inputs[0]->attr.dim_num,
                shape, &new_rank );

        vsi_nn_kernel_param_add_float32( param, "min_value",  min_value );
        vsi_nn_kernel_param_add_float32( param, "max_value",  max_value );

        if( ret )
        {
            reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                    inputs[0], shape, new_rank );
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                    outputs[0], shape, new_rank );

            self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                    "clip",
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
    BEGIN_IO_TYPE_DECL(CLIP, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_F32)
        IO_TYPE(D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_BF16,       D_BF16)
    END_IO_TYPE_DECL(CLIP)
    if(!VALIDATE_OP_IO_TYPES(CLIP, self, inputs, self->input.num, outputs, self->output.num)) {
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
    float min = self->nn_param.clip.min;
    float max = self->nn_param.clip.max;

    for (i = 0; i < _VSI_NN_CLIP_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.clip.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.clip.local.local_tensor[i]));
            self->nn_param.clip.local.local_tensor[i] = NULL;
        }
    }

    if (self->nn_param.clip.local2 != NULL)
    {
        free(self->nn_param.clip.local2);
        self->nn_param.clip.local2 = NULL;
    }

    if ( (min == -1.0f && max == 1.0f)
      || (min == 0.0f && max == 6.0f) )
    {
        vsi_nn_internal_deinit_node_wksp( self );
    }

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    self->nn_param.clip.local2   =
    (vsi_nn_clip_lcl2_data *)malloc(sizeof(vsi_nn_clip_lcl2_data));
    if (NULL == self->nn_param.reduce.local2)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(self->nn_param.clip.local2, 0, sizeof(vsi_nn_clip_lcl2_data));
    return status;
} /* op_init() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_internal_node_t* curr = NULL;
    float min = self->nn_param.clip.min;
    float max = self->nn_param.clip.max;

    if ( (min == -1.0f && max == 1.0f)
      || (min == 0.0f && max == 6.0f) )
    {
        vsi_nn_internal_init_node_wksp(self);
        if (min == -1.0f && max == 1.0f)
        {
            curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU1, 0, 0);
        }
        else
        {
            curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU6, 0, 0);
        }
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];

        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        ret = vsi_nn_op_common_setup(self, inputs, outputs);
    }
    return ret;
} /* op_init() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CLIP,
    /* init       */ op_init,
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

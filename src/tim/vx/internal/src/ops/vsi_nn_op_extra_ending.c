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
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

extern vx_kernel_description_t * vx_kernel_EXTRA_ENDING_list[];

static void check_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index,
    vx_bool rsFlg
    )
{
    vsi_nn_tensor_attr_t attr;

    if (index == 0)
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.extra_ending.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.extra_ending.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if (index == 1)
    {
        params[index] = (vx_reference)input->t;
    }
    else if (index == 2)
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.extra_ending.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.extra_ending.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else
    {
        VSILOGE("No more local tensor!(pow) at [%s : %d]\n", __FILE__, __LINE__);
    }
}

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_tensor_t* extraInput
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[3];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, 0);
    check_tensor_shape(self, extraInput, params, 1, 0);
    check_tensor_shape(self, outputs[0], params, 2, 0);

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 3 );

    return status;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_tensor_t* extraInput
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[3];
    vx_border_t border;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, 0);
    check_tensor_shape(self, extraInput, params, 1, 0);
    check_tensor_shape(self, outputs[0], params, 2, 0);
    /*TODO: Add code if need to change your parameter*/

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 3 );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e outputDataFormat    = outputs[0]->attr.dtype.vx_type;

    if (outputDataFormat == VSI_NN_TYPE_INT16 || outputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 1;
    }
    if (outputDataFormat == VSI_NN_TYPE_INT8)
    {
        kernel_info->kernel_index = 2;
    }
    if (outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info->kernel_index = 3;
    }
    else
    {
        VSILOGE("Not support input or output data format!(extra ending) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    return VSI_SUCCESS;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_tensor_t* tmpRealInput = NULL;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_EXTRA_ENDING_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_extra_ending";

    {
        vsi_nn_tensor_attr_t attr;

        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = self->nn_param.extra_ending.length;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 2;
        attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
        attr.vtl = FALSE;
        tmpRealInput = vsi_nn_CreateTensorFromData(self->graph,
                (uint8_t*)&self->nn_param.extra_ending.value, &attr);
    }

    if( kernel_info.type == VX_KERNEL_TYPE_VX)
    {
        kernel_info.kernel_index = 1;
        kernel_info.init_index = 1;
        vx_op_pre_compute(self, inputs, outputs, &kernel_info);
    }
    else /*kernel_info.type = VX_KERNEL_TYPE_CPU;*/
    {
        kernel_info.kernel_index = 0;
        kernel_info.init_index = 0;
        kernel_info.type = VX_KERNEL_TYPE_CPU;
    }

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
    if (kernel_info.resource_name)
    {
        free(kernel_info.resource_name);
    }
    if( NULL == self->n )
    {
        status = VSI_FAILURE;
        goto final;
    }

    if(kernel_info.type == VX_KERNEL_TYPE_VX)
    {
        status = vx_op_compute(self, inputs, outputs, tmpRealInput);
    }
    else
    {
        status = cpu_op_compute(self, inputs, outputs, tmpRealInput);
    }

final:
    if(tmpRealInput) vsi_nn_ReleaseTensor(&tmpRealInput);
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(EXTRA_ENDING, 1, 1)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
    END_IO_TYPE_DECL(EXTRA_ENDING)
    if (!VALIDATE_OP_IO_TYPES(EXTRA_ENDING, self, inputs, self->input.num, outputs, self->output.num))
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
    /* TODO: Add code to comput outputs' shape. */
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        VSILOGE("output size cannot be zero!(EXTRA_ENDING)\n");
        return FALSE;
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_EXTRA_ENDING_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.extra_ending.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.extra_ending.local.local_tensor[i]));
            self->nn_param.extra_ending.local.local_tensor[i] = NULL;
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
    /* op_name    */ EXTRA_ENDING,
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

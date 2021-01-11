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
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (0)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_TENSORSTACKCONCAT_list[];

static vsi_bool _reshape_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i = 0;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t axis = 0;
    vsi_nn_tensorstackconcat_param * p = NULL;
    uint32_t before_size = 1;
    uint32_t after_size = 1;
    uint32_t * input_sizes = inputs[0]->attr.size;
    uint32_t dims = inputs[0]->attr.dim_num;
    uint32_t * output_sizes = outputs[0]->attr.size;
    uint32_t new_dims = 0;

    p = &(self->nn_param.tensorstackconcat);
    axis = p->axis;

    for ( i = 0; i < (uint32_t)axis; i++)
    {
        before_size *= input_sizes[i];
    }
    for ( i = axis + 1; i < dims; i++)
    {
        after_size *= input_sizes[i];
    }
    sizes[0] = before_size;
    sizes[1] = input_sizes[axis];
    sizes[2] = after_size;
    new_dims = 3;
    p->local->local_tensor[0] = vxReshapeTensor(inputs[0]->t, (int32_t *)sizes, new_dims);

    sizes[0] = 1;
    sizes[1] = 1;
    new_dims = 2;
    p->local->local_tensor[1] = vxReshapeTensor(inputs[1]->t, (int32_t *)sizes, new_dims);

    before_size = 1;
    after_size = 1;
    for ( i = 0; i < (uint32_t)axis; i++)
    {
        before_size *= output_sizes[i];
    }
    for ( i = axis + 1; i < dims; i++)
    {
        after_size *= output_sizes[i];
    }
    sizes[0] = before_size;
    sizes[1] = output_sizes[axis];
    sizes[2] = after_size;
    new_dims = 3;
    p->local->local_tensor[2] = vxReshapeTensor(outputs[0]->t, (int32_t *)sizes, new_dims);

    p->axis = 1;
    return TRUE;
}

static void _set_inputs_outputs
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensorstackconcat_param *p = NULL;
    uint32_t i = 0;

    p = &(self->nn_param.tensorstackconcat);

    for (i = 0; i < _IO_NUM; i++)
    {
        params[i] = (vx_reference)(p->local->local_tensor[i]);
    }
} /* _set_inputs_outputs() */

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    //_release_params( args, _ARG_NUM );

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
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputDataFormat    = outputs[0]->attr.dtype.vx_type;
    int8_t        inputFixedPointPos  = inputs[0]->attr.dtype.fl;
    int8_t        outputFixedPointPos = outputs[0]->attr.dtype.fl;
    int32_t       inputZeroPoint      = inputs[0]->attr.dtype.zero_point;
    int32_t       outputZeroPoint     = outputs[0]->attr.dtype.zero_point;
    vx_float32    inputScale          = inputs[0]->attr.dtype.scale;
    vx_float32    outputScale         = outputs[0]->attr.dtype.scale;
    vsi_bool      is16Bits            = FALSE;
    vsi_bool      is8Bits             = FALSE;

    is16Bits = ((inputDataFormat == VSI_NN_TYPE_FLOAT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
            || (inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_INT16
            && inputFixedPointPos == outputFixedPointPos)) ? TRUE : FALSE;
    is8Bits = ((inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_INT8
            && inputFixedPointPos == outputFixedPointPos)
            || (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8
            && inputZeroPoint == outputZeroPoint && inputScale == outputScale)) ? TRUE : FALSE;

    if (is16Bits)
    {
        kernel_info->kernel_index = 1;
    }
    else if (is8Bits)
    {
        kernel_info->kernel_index = 2;
    }
    else
    {
        VSILOGE("Not support input or output data format!(TENSORSTACKCONCAT) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    return VSI_SUCCESS;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_border_t border;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    NULL
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));

    /* reshape input/output */
    _reshape_tensor( self, inputs, outputs);

    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_tensorstackconcat";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_TENSORSTACKCONCAT_list;
    kernel_info.init_index = 1;

    if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
    {
        vx_op_pre_compute(self, inputs, outputs, &kernel_info);
    }

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
    if (kernel_info.resource_name) free(kernel_info.resource_name);
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
    }
    return status;
} /* op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    vsi_nn_tensorstackconcat_param *p = NULL;
    vsi_nn_stackconcat_lcl_data *local = NULL;
    int32_t axis = 0;

    if( NULL == self )
    {
        return ret;
    }

    p = &(self->nn_param.tensorstackconcat);
    axis = p->axis;
    local = (vsi_nn_stackconcat_lcl_data *)malloc(sizeof(vsi_nn_stackconcat_lcl_data));
    if (NULL == local)
    {
        return ret;
    }
    memset(local, 0, sizeof(vsi_nn_stackconcat_lcl_data));
    p->local = local;

    if (axis < 0)
    {
        axis = axis + inputs[0]->attr.dim_num;
        p->axis = axis;
    }

    return TRUE;
} /* op_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensorstackconcat_param *p = NULL;
    int32_t axis = 0;
    int32_t dims = (int32_t)inputs[0]->attr.dim_num;
    int32_t out_dims = (int32_t)outputs[0]->attr.dim_num;

    p = &(self->nn_param.tensorstackconcat);
    axis = p->axis;

    if (axis < 0)
    {
        axis = axis + dims;
    }

    if (axis > (dims - 1))
    {
        VSILOGE("Invalid Axis: %d, (TENSORSTACKCONCAT) at [%s : %d]\n", axis, __FILE__, __LINE__);
        return FALSE;
    }
    if( VSI_NN_DIM_AUTO == out_dims )
    {
        VSILOGE("Invalid output, (TENSORSTACKCONCAT) at [%s : %d]\n", __FILE__, __LINE__);
        return FALSE;
    }
    if( dims != out_dims )
    {
        VSILOGE("Input and output's dims not matched, (TENSORSTACKCONCAT) at [%s : %d]\n", __FILE__, __LINE__);
        return FALSE;
    }

    {
        BEGIN_IO_TYPE_DECL(TENSORSTACKCONCAT, 2, 1)
            IO_TYPE(D_F16, D_F16, D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I8|Q_DFP)
        END_IO_TYPE_DECL(TENSORSTACKCONCAT)
        if(!VALIDATE_OP_IO_TYPES(TENSORSTACKCONCAT, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return TRUE;
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.tensorstackconcat.axis = 1;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensorstackconcat_param *p = &(self->nn_param.tensorstackconcat);
    uint32_t i = 0;
    if (p->local)
    {
        for (i = 0; i < _VSI_NN_STACKCONCAT_LOCAL_TENSOR_NUM; i++)
        {
            if (p->local->local_tensor[i])
            {
                vxReleaseTensor(&(p->local->local_tensor[i]));
                p->local->local_tensor[i] = NULL;
            }
        }
        vsi_nn_safe_free(p->local);
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
    /* op_name    */ TENSORSTACKCONCAT,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif


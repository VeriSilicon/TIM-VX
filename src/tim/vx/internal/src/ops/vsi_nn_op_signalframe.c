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

#define _ARG_NUM            (5)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define ENABLE_CPU 0
#define TENSOR_ALL 0

extern vx_kernel_description_t * vx_kernel_SIGNALFRAME_list[];

static vsi_status _create_local_tensor
    (
    vsi_nn_node_t * self
    )
{
    //vsi_nn_tensor_t *signal_tensor = NULL;
    //vsi_nn_tensor_t *frame_tensor = NULL;
    vsi_nn_tensor_t *window_length_tensor = NULL;
    vsi_nn_tensor_t *step_tensor = NULL;
    vsi_nn_tensor_t *pad_end_tensor = NULL;
    vsi_nn_tensor_t *pad_tensor = NULL;
    vsi_nn_tensor_t *axis_tensor = NULL;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    window_length_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.signalframe.window_length,
        VSI_NN_TYPE_UINT32);
    if(NULL == window_length_tensor)
    {
        goto error;
    }

    step_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.signalframe.step,
        VSI_NN_TYPE_UINT32);
    if(NULL == step_tensor)
    {
        goto error;
    }

    pad_end_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.signalframe.pad_end,
        VSI_NN_TYPE_UINT32);
    if(NULL == pad_end_tensor)
    {
        goto error;
    }

    pad_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.signalframe.pad,
        VSI_NN_TYPE_UINT32);
    if(NULL == pad_tensor)
    {
        goto error;
    }

    axis_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.signalframe.axis,
        VSI_NN_TYPE_UINT32);
    if(NULL == axis_tensor)
    {
        goto error;
    }

    self->nn_param.signalframe.local.window_length_tensor = window_length_tensor;
    self->nn_param.signalframe.local.step_tensor = step_tensor;
    self->nn_param.signalframe.local.pad_end_tensor = pad_end_tensor;
    self->nn_param.signalframe.local.pad_tensor = pad_tensor;
    self->nn_param.signalframe.local.axis_tensor = axis_tensor;

    return VSI_SUCCESS;
error:
    if(window_length_tensor)vsi_nn_ReleaseTensor(&window_length_tensor);
    if(step_tensor)vsi_nn_ReleaseTensor(&step_tensor);
    if(pad_end_tensor)vsi_nn_ReleaseTensor(&pad_end_tensor);
    if(pad_tensor)vsi_nn_ReleaseTensor(&pad_tensor);
    if(axis_tensor)vsi_nn_ReleaseTensor(&axis_tensor);
    return VSI_FAILURE;
} /* _create_local_tensor() */

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

    if (index == 0 )
    {
        if( input->attr.dim_num == 1 )
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.signalframe.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if(index == 1 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.signalframe.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 4)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[2] *= attr.size[3];
            attr.size[3] = 1;
            attr.dim_num = 3;
            self->nn_param.signalframe.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;

    }
    else
    {
        VSILOGE("No more local tensor!(signalframe) at [%s : %d]\n", __FILE__, __LINE__);
    }
}

static void check_local_tensor_shape
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    uint32_t index,
    vx_bool rsFlg
    )
{
    vsi_nn_tensor_attr_t attr;

    if( self->nn_param.signalframe.local.window_length_tensor->attr.dim_num == 1 )
    {
        memcpy(&attr, &(self->nn_param.signalframe.local.window_length_tensor->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.dim_num = 2;
        self->nn_param.signalframe.local.local_tensor[index] =
            vxReshapeTensor(self->nn_param.signalframe.local.window_length_tensor->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
    }
    else
        params[index] = (vx_reference)self->nn_param.signalframe.local.window_length_tensor->t;
    index++;

    if( self->nn_param.signalframe.local.step_tensor->attr.dim_num == 1 )
    {
        memcpy(&attr, &(self->nn_param.signalframe.local.step_tensor->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.dim_num = 2;
        self->nn_param.signalframe.local.local_tensor[index] =
            vxReshapeTensor(self->nn_param.signalframe.local.step_tensor->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
    }
    else
        params[index] = (vx_reference)self->nn_param.signalframe.local.step_tensor->t;
    index++;

    if( self->nn_param.signalframe.local.pad_end_tensor->attr.dim_num == 1 )
    {
        memcpy(&attr, &(self->nn_param.signalframe.local.pad_end_tensor->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.dim_num = 2;
        self->nn_param.signalframe.local.local_tensor[index] =
            vxReshapeTensor(self->nn_param.signalframe.local.pad_end_tensor->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
    }
    else
        params[index] = (vx_reference)self->nn_param.signalframe.local.pad_end_tensor->t;
    index++;

    if( self->nn_param.signalframe.local.pad_tensor->attr.dim_num == 1 )
    {
        memcpy(&attr, &(self->nn_param.signalframe.local.pad_tensor->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.dim_num = 2;
        self->nn_param.signalframe.local.local_tensor[index] =
            vxReshapeTensor(self->nn_param.signalframe.local.pad_tensor->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
    }
    else
        params[index] = (vx_reference)self->nn_param.signalframe.local.pad_tensor->t;
    index++;

    if( self->nn_param.signalframe.local.axis_tensor->attr.dim_num == 1 )
    {
        memcpy(&attr, &(self->nn_param.signalframe.local.axis_tensor->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.dim_num = 2;
        self->nn_param.signalframe.local.local_tensor[index] =
            vxReshapeTensor(self->nn_param.signalframe.local.axis_tensor->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.signalframe.local.local_tensor[index];
    }
    else
        params[index] = (vx_reference)self->nn_param.signalframe.local.axis_tensor->t;

}

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_signalframe_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.signalframe);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_UINT32, window_length );
    _SET_PARAM( 1, VX_TYPE_UINT32, step );
    _SET_PARAM( 2, VX_TYPE_UINT32, pad_end );
    _SET_PARAM( 3, VX_TYPE_UINT32, pad );
    _SET_PARAM( 4, VX_TYPE_UINT32, axis );
#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    uint32_t num
    )
{
    uint32_t i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */

#if ENABLE_CPU
static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }

    /*for( i = 0; i < _ARG_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i + 1]->t;
    }*/
} /* _set_inputs_outputs() */
#endif

#if ENABLE_CPU
static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_bool rsFlg = FALSE;
    vx_reference * args;
    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, outputs[0], params, 1, rsFlg);
    if(TENSOR_ALL)
        check_local_tensor_shape(self, params, 2, rsFlg);
    else
        _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}
#endif

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
    uint32_t axis0 = self->nn_param.signalframe.axis;
    uint32_t axis = axis0;
    uint32_t dim = inputs[0]->attr.dim_num;
    vx_bool dataTypeFlg = FALSE;
    vx_bool etFlg = FALSE;

    if((inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_INT8) ||
        (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8))
        etFlg = TRUE;

    if ((inputDataFormat == VSI_NN_TYPE_FLOAT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16) ||
        (inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_INT16) ||
        etFlg)
        dataTypeFlg = TRUE;

    axis = dim - axis0 - 1;

    if (dataTypeFlg
        && ((dim == 1 && axis==0) || (dim == 2 && axis==1) || (dim == 3 && axis==2)))
    {
        kernel_info->kernel_index = 1;
        if(etFlg)
        {
            kernel_info->kernel_index = 4;
        }
    }
    else if(dataTypeFlg
        && ((dim == 2 && axis==0) || (dim == 3 && axis==1)))
    {
        kernel_info->kernel_index = 2;
        if(etFlg)
        {
            kernel_info->kernel_index = 5;
        }
    }
    else if(dataTypeFlg
        && (dim == 3 && axis==0))
    {
        kernel_info->kernel_index = 3;
        if(etFlg)
        {
            kernel_info->kernel_index = 6;
        }
    }
    else
    {
        VSILOGE("Not support input or output data format!(SIGNALFRAME) at [%s : %d]\n", __FILE__, __LINE__);
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
    vx_bool rsFlg = FALSE;
    vx_reference * args;
    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, outputs[0], params, 1, rsFlg);
    if(TENSOR_ALL)
        check_local_tensor_shape(self, params, 2, rsFlg);
    else
        _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status |= vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
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
    status = VSI_SUCCESS;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    if(0)
    {
        status = _create_local_tensor(self);
        if(status != VSI_SUCCESS)
        {
            return status;
        }
    }

#if ENABLE_CPU //cpu
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_signalframe";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_SIGNALFRAME_list;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            return VSI_FAILURE;
        }

        status = cpu_op_compute(self, inputs, outputs);

        return status;
    }
#endif

    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_signalframe";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_SIGNALFRAME_list;
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

    status |= vx_op_compute(self, inputs, outputs);

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(SIGNAL_FRAME, 1, 1)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_F32, D_F32)
    END_IO_TYPE_DECL(SIGNAL_FRAME)
    if(!VALIDATE_OP_IO_TYPES(SIGNAL_FRAME, self, inputs, self->input.num, outputs, self->output.num)) {
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
    for (i = 0; i < _VSI_NN_SIGNALFRAME_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.signalframe.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.signalframe.local.local_tensor[i]));
            self->nn_param.signalframe.local.local_tensor[i] = NULL;
        }
    }

    if(self->nn_param.signalframe.local.window_length_tensor)
        vsi_nn_ReleaseTensor(&self->nn_param.signalframe.local.window_length_tensor);
    if(self->nn_param.signalframe.local.step_tensor)
        vsi_nn_ReleaseTensor(&self->nn_param.signalframe.local.step_tensor);
    if(self->nn_param.signalframe.local.pad_end_tensor)
        vsi_nn_ReleaseTensor(&self->nn_param.signalframe.local.pad_end_tensor);
    if(self->nn_param.signalframe.local.pad_tensor)
        vsi_nn_ReleaseTensor(&self->nn_param.signalframe.local.pad_tensor);
    if(self->nn_param.signalframe.local.axis_tensor)
        vsi_nn_ReleaseTensor(&self->nn_param.signalframe.local.axis_tensor);
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t         i;
    vsi_bool        ret;
    uint32_t        axis;

    ret = TRUE;
    if( VSI_NN_DIM_AUTO != outputs[0]->attr.dim_num )
    {
        return ret;
    }

    axis = self->nn_param.signalframe.axis;
    if(axis >= inputs[0]->attr.dim_num)
    {
        return FALSE;
    }

    /* signal frame will increase dim num */
    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num + 1;
    for(i = 0; i < axis; i++)
    {
        outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
    }
    if(self->nn_param.signalframe.pad_end)
    {
        outputs[0]->attr.size[axis] = inputs[0]->attr.size[axis];
    }
    else
    {
        if(inputs[0]->attr.size[axis] >= self->nn_param.signalframe.window_length)
        {
            outputs[0]->attr.size[axis] = (inputs[0]->attr.size[axis] - self->nn_param.signalframe.window_length) \
                / self->nn_param.signalframe.step + 1;
        }
        else
        {
            outputs[0]->attr.size[axis] = 0;
            return FALSE;
        }
    }
    for(i = axis; i < inputs[0]->attr.dim_num; i++)
    {
        outputs[0]->attr.size[i + 1] = inputs[0]->attr.size[i];
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SIGNAL_FRAME,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif


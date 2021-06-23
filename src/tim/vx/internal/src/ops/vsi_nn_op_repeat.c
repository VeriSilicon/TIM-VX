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
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status _create_local_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t* repeat_host = self->nn_param.repeat.repeat_host;
    int32_t  axis = self->nn_param.repeat.axis;
    vsi_nn_repeat_lcl_data *local = self->nn_param.repeat.local;
    uint32_t shape[VSI_NN_MAX_DIM_NUM] = {1, 1, 1, 1};
    uint32_t i = 0;

    if (axis == -1)
    {
        axis = 0;
        for(i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            shape[0] *= inputs[0]->attr.size[i];
        }

        local->reshaped_input = vsi_nn_reshape_tensor(self->graph, inputs[0], shape, 1);

        shape[0] = 1;
        for(i = 0; i < outputs[0]->attr.dim_num; i++)
        {
            shape[0] *= outputs[0]->attr.size[i];
        }
        local->reshaped_output = vsi_nn_reshape_tensor(self->graph, outputs[0], shape, 1);
    }

    if (repeat_host)
    {
        vsi_nn_tensor_attr_t attr;
        int32_t len = 0;

        if (self->nn_param.repeat.axis < 0)
        {
            len = local->reshaped_input->attr.size[0];
        }
        else if (axis == 1 || inputs[0]->attr.dim_num == 1)
        {
            len = inputs[0]->attr.size[0];
        }
        else if (axis == 0)
        {
            len = inputs[0]->attr.size[1];
        }
        else if (axis == 2)
        {
            len = inputs[0]->attr.size[2];
        }

        memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.is_const = FALSE;
        attr.vtl = TRUE;
        attr.size[0] = len;
        attr.size[1] = 1;
        attr.dim_num = 2;

        local->repeat_tensor = vsi_nn_CreateTensorFromData(self->graph, (uint8_t*)repeat_host, &attr);
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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_kernel_node_t n = NULL;
    int32_t  axis = self->nn_param.repeat.axis;
    vsi_nn_tensor_t * tmp_inputs[2]  = {NULL, NULL};
    vsi_nn_tensor_t * tmp_output[1]  = {NULL};
    vsi_nn_repeat_lcl_data *local = self->nn_param.repeat.local;

    status = _create_local_tensor(self, inputs, outputs);
    if (status != VSI_SUCCESS)
    {
        VSILOGE("Create local tensor fail");
        return status;
    }

    if (local->reshaped_input)
    {
        tmp_inputs[0] = local->reshaped_input;
        tmp_output[0] = local->reshaped_output;
    }
    else
    {
        tmp_inputs[0] = inputs[0];
        tmp_output[0] = outputs[0];
    }

    if (local->repeat_tensor)
    {
        tmp_inputs[1] = local->repeat_tensor;
    }
    else
    {
        tmp_inputs[1] = inputs[1];
    }

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "axis", axis );
    n = vsi_nn_kernel_selector( self->graph, "repeat",
                    tmp_inputs, _INPUT_NUM, tmp_output, _OUTPUT_NUM, param );
    if ( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
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
    vsi_nn_repeat_param * p = NULL;

    BEGIN_IO_TYPE_DECL(REPEAT, 2, 1)
        IO_TYPE(D_F16,  D_I32,  D_F16)
        IO_TYPE(D_F32,  D_I32,  D_F32)
        IO_TYPE(D_I32,  D_I32,  D_I32)
        IO_TYPE(D_I8,   D_I32,  D_I8)
        IO_TYPE(D_U8,   D_I32,  D_U8)
        IO_TYPE(D_I16,  D_I32,  D_I16)
        IO_TYPE(D_I8|Q_DFP,  D_I32,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_I32,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP, D_I32,  D_I16|Q_DFP)
    END_IO_TYPE_DECL(REPEAT)
    if (!VALIDATE_OP_IO_TYPES(REPEAT, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    p = (vsi_nn_repeat_param *)&(self->nn_param.repeat);
    if ((p->repeat_host == NULL && p->maxlen < 1) || p->axis > 3)
    {
        VSILOGE("Unsupported parameters");
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
    vsi_nn_repeat_param * p = NULL;
    int32_t i = 0;
    int32_t sum = 0;
    int32_t axis = 0;
    p = (vsi_nn_repeat_param *)&(self->nn_param.repeat);
    axis = p->axis;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        for(i = 0; i < (int32_t)inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }

        if (p->repeat_host)
        {
            for(i = 0; i < p->repeat_len; i++)
            {
                sum += p->repeat_host[i];
            }
        }
        else
        {
            sum = p->maxlen;
        }

        if (inputs[0]->attr.dim_num == 1 || axis == -1 || axis == 1)
        {
            outputs[0]->attr.size[0] = sum;
        }
        else if (axis == 0)
        {
            outputs[0]->attr.size[1] = sum;
        }
        else if (axis == 2)
        {
            outputs[0]->attr.size[2] = sum;
        }
        else if (axis == 3)
        {
            outputs[0]->attr.size[3] = sum;
        }
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.repeat.local =
    (vsi_nn_repeat_lcl_data *)malloc(sizeof(vsi_nn_repeat_lcl_data));
    if (NULL == self->nn_param.repeat.local)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset( self->nn_param.repeat.local, 0, sizeof(vsi_nn_repeat_lcl_data) );

    self->nn_param.repeat.local->reshaped_input = NULL;
    self->nn_param.repeat.local->reshaped_output = NULL;
    self->nn_param.repeat.local->repeat_tensor = NULL;
    self->nn_param.repeat.repeat_host = NULL;
    self->nn_param.repeat.repeat_len = 0;
    self->nn_param.repeat.axis = -1;
    self->nn_param.repeat.maxlen = -1;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_repeat_param *p = &(self->nn_param.repeat);
    if (p->local->reshaped_input)
    {
        vsi_nn_ReleaseTensor(&(p->local->reshaped_input));
        p->local->reshaped_input = NULL;
    }
    if (p->local->reshaped_output)
    {
        vsi_nn_ReleaseTensor(&(p->local->reshaped_output));
        p->local->reshaped_output = NULL;
    }
    if (p->local->repeat_tensor)
    {
        vsi_nn_ReleaseTensor(&(p->local->repeat_tensor));
        p->local->repeat_tensor = NULL;
    }
    if (self->nn_param.repeat.local)
    {
        free(self->nn_param.repeat.local);
        self->nn_param.repeat.local = NULL;
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
    /* op_name    */ REPEAT,
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


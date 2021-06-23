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
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status _create_local_tensor
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensor_t *rank_tensor = NULL;
    vsi_nn_tensor_t *act_tensor = NULL;

    /* activation must set to 0, so the sdk will call VX_NN_ACTIVATION_NONE */
    int32_t activation = 0;

    if(NULL == self)
    {
        goto error;
    }

    act_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&activation,
        VSI_NN_TYPE_INT32);
    if(NULL == act_tensor)
    {
        goto error;
    }

    rank_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.svdf.rank,
        VSI_NN_TYPE_INT32);
    if(NULL == act_tensor)
    {
        goto error;
    }

    self->nn_param.svdf.local.act_tensor = act_tensor;
    self->nn_param.svdf.local.rank_tensor = rank_tensor;
    return VSI_SUCCESS;
error:
    if(rank_tensor)vsi_nn_ReleaseTensor(&rank_tensor);
    if(act_tensor)vsi_nn_ReleaseTensor(&act_tensor);
    return VSI_FAILURE;
} /* _create_local_tensor() */

static vsi_status _init_svdf_param
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vx_nn_svdf_params_t *param
    )
{
    vsi_nn_svdf_param *p = &self->nn_param.svdf;

    param->state_in        = REQUIRED_IO(inputs[1]);
    param->weights_feature = REQUIRED_IO(inputs[2]);
    param->recurrent_time  = REQUIRED_IO(inputs[3]);
    param->bias            = OPTIONAL_IO(inputs[4]);
    param->activation      = REQUIRED_IO(p->local.act_tensor);
    param->rank            = REQUIRED_IO(p->local.rank_tensor);

    return VSI_SUCCESS;
} /* _init_svdf_param() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_svdf_params_t param;
    vsi_nn_tensor_t * bias_tensor = NULL;

    status = VSI_FAILURE;
    memset(&param, 0, sizeof(param));

    status = _create_local_tensor(self);
    if(VSI_SUCCESS != status)
    {
        return status;
    }

    status = _init_svdf_param(self, inputs, &param);
    if(VSI_SUCCESS != status)
    {
        return status;
    }

    if (param.bias == NULL)
    {
        vsi_nn_tensor_attr_t attr;
        int32_t count = inputs[2]->attr.size[1];

        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = count;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        bias_tensor = vsi_nn_CreateTensor(self->graph, &attr);
        param.bias = bias_tensor->t;
    }

    self->n = vxSVDFLayer(
        self->graph->g,
        REQUIRED_IO(inputs[0]),
        &param,
        sizeof(param),
        REQUIRED_IO(outputs[1]), /* state out */
        REQUIRED_IO(outputs[0])
        );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    if (bias_tensor != NULL) vsi_nn_ReleaseTensor(&bias_tensor);
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    if(2 != inputs[0]->attr.dim_num)
    {
        VSILOGE("SVDF input dimension should be 2");
        ret = FALSE;
    }

    if (ret)
    {
        BEGIN_IO_TYPE_DECL(SVDF, 5, 2)
            IO_TYPE(D_F16, D_F16, D_F16, D_F16, D_F16,  D_F16, D_F16)
            IO_TYPE(D_F16, D_F16, D_F16, D_F16, D_F32,  D_F16, D_F16)
            IO_TYPE(D_F32, D_F16, D_F16, D_F16, D_F32,  D_F32, D_F16)
            IO_TYPE(D_F32, D_F32, D_F32, D_F32, D_F32,  D_F32, D_F32)
            IO_TYPE(D_F16, D_F16, D_F16, D_F16, D_NONE, D_F16, D_F16)
            IO_TYPE(D_F16, D_F16, D_F16, D_F16, D_NONE, D_F32, D_F16)
            IO_TYPE(D_F32, D_F16, D_F16, D_F16, D_NONE, D_F32, D_F32)
            IO_TYPE(D_F32, D_F32, D_F32, D_F32, D_NONE, D_F32, D_F32)
        END_IO_TYPE_DECL(SVDF)
        if(!VALIDATE_OP_IO_TYPES(SVDF, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return ret;
} /* op_check() */

/*
 * input[0]: input
 * input[1]: state_in (variable)
 * input[2]: weights_feature
 * input[3]: weights_time
 * input[4]: bias
 * output[0]: output
 * output[1]: state_out
 */
static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_svdf_param *p;

    p = (vsi_nn_svdf_param *)&self->nn_param.svdf;
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = p->num_units;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
    }
    if(VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num)
    {
        outputs[1]->attr.dim_num = inputs[1]->attr.dim_num;
        outputs[1]->attr.size[0] = inputs[1]->attr.size[0];
        outputs[1]->attr.size[1] = inputs[1]->attr.size[1];
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensor_t *act_tensor, *rank_tensor;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    act_tensor = self->nn_param.svdf.local.act_tensor;
    rank_tensor = self->nn_param.svdf.local.rank_tensor;
    if(NULL != self->n)
    {
        if(act_tensor)vsi_nn_ReleaseTensor(&act_tensor);
        if(rank_tensor)vsi_nn_ReleaseTensor(&rank_tensor);
        vxReleaseNode( &self->n );
        self->n = NULL;
    }

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SVDF,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 5,
    /* output_num */ 2
    );
#ifdef __cplusplus
}
#endif

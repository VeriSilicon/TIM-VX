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
#include "utils/vsi_nn_tensor_op.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"

#define _INPUT_NUM          (LSTMUNIT_ACT_INPUTS_COUNT)
#define _OUTPUT_NUM         (LSTMUNIT_ACT_OUTUTS_COUNT)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    int32_t _is_ln= 0;
    int32_t _is_cifg= 0;
    int32_t _is_proj= 0;
    int32_t _is_hybrid= 0;
    int32_t _is_peephole= 0;
    int32_t recurrent_activation;
    float    cell_clip;
    float    proj_clip;
    float    forget_bias;
    vsi_nn_lstmunit_activation_param * p = NULL;

    p                    = &(self->nn_param.lstmunit_activation);
    _is_ln               = p->is_layer_norm ? 1 : 0;
    _is_cifg             = p->is_cifg ? 1 : 0;
    _is_proj             = p->is_projection ? 1 : 0;
    _is_hybrid           = p->is_hybrid ? 1 : 0;
    _is_peephole         = p->is_peephole ? 1 : 0;
    recurrent_activation = (int32_t)(p->recurrent_activation);
    cell_clip            = p->cell_clip;
    proj_clip            = p->proj_clip;
    forget_bias          = p->forget_bias;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "_is_ln",       _is_ln );
    vsi_nn_kernel_param_add_int32( param, "_is_cifg",     _is_cifg );
    vsi_nn_kernel_param_add_int32( param, "_is_proj",     _is_proj );
    vsi_nn_kernel_param_add_int32( param, "_is_hybrid",   _is_hybrid );
    vsi_nn_kernel_param_add_int32( param, "_is_peephole", _is_peephole );
    vsi_nn_kernel_param_add_int32( param, "recurrent_activation", recurrent_activation );
    vsi_nn_kernel_param_add_float32( param, "cell_clip" ,  cell_clip );
    vsi_nn_kernel_param_add_float32( param, "proj_clip" ,  proj_clip );
    vsi_nn_kernel_param_add_float32( param, "forget_bias", forget_bias );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                    "lstmunit_activation",
                    inputs,  _INPUT_NUM,
                    outputs, _OUTPUT_NUM, param );

    vsi_nn_kernel_param_release( &param );

    if( self->n )
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
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_lstmunit_activation_param * p;
    vsi_nn_dtype_t dst_dtype;
    int32_t ifco_start_index = 0;
    vsi_nn_tensor_attr_t attr;
    int32_t i = 0;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if ( NULL == self )
    {
        return FALSE;
    }

    p = &(self->nn_param.lstmunit_activation);

    p->is_cifg = inputs[LSTMUNIT_ACT_INPUT_FC_I] == NULL;
    p->is_projection = outputs[LSTMUNIT_ACT_HSTATE_OUT] == NULL;
    if (self->graph->ctx->config.support_stream_processor)
    {
        p->is_layer_norm = inputs[LSTMUNIT_ACT_HSTATE_FC_F] == NULL;
    }
    else
    {
        p->is_layer_norm = inputs[LSTMUNIT_ACT_LN_WF] != NULL;
    }
    p->is_hybrid = p->is_layer_norm ? 0 : inputs[LSTMUNIT_ACT_DATA_BF] != NULL;
    p->recurrent_activation = p->recurrent_activation == VSI_NN_ACT_NONE ?
        VSI_NN_ACT_SIGMOID : p->recurrent_activation;

    for( i = ifco_start_index; i < 4; i++ )
    {
        vsi_nn_tensor_t* t0 = NULL;
        vsi_nn_tensor_t* t1 = NULL;
        dst_dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        dst_dtype.vx_type = VSI_NN_TYPE_FLOAT32;

        if (inputs[LSTMUNIT_ACT_DATA_BI + i] && inputs[LSTMUNIT_ACT_DATA_BI + i]->attr.dim_num == 1)
        {
            memcpy(&attr, &(inputs[LSTMUNIT_ACT_DATA_BI + i]->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            t0 = vsi_nn_reshape_tensor(self->graph, inputs[LSTMUNIT_ACT_DATA_BI + i], attr.size, attr.dim_num);
            CHECK_PTR_FAIL_GOTO( t0, "create tensor fail.", final );

            if ( dst_dtype.vx_type != t0->attr.dtype.vx_type
                && dst_dtype.qnt_type != t0->attr.dtype.qnt_type )
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_BI + i] =
                    vsi_nn_ConvertTensorDtype( self->graph, t0, &dst_dtype );

                vsi_safe_release_tensor(t0);
            }
            else
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_BI + i] = t0;
            }

            inputs[LSTMUNIT_ACT_DATA_BI + i] = p->local.tensors[LSTMUNIT_ACT_TENSOR_BI + i];
        }

        if (inputs[LSTMUNIT_ACT_LN_WI + i] && inputs[LSTMUNIT_ACT_LN_WI + i]->attr.dim_num == 1)
        {
            memcpy(&attr, &(inputs[LSTMUNIT_ACT_LN_WI + i]->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            t1 = vsi_nn_reshape_tensor(self->graph, inputs[LSTMUNIT_ACT_LN_WI + i], attr.size, attr.dim_num);
            CHECK_PTR_FAIL_GOTO( t1, "create tensor fail.", final );

            if ( dst_dtype.vx_type != t1->attr.dtype.vx_type
                && dst_dtype.qnt_type != t1->attr.dtype.qnt_type )
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_LN_WI + i] =
                    vsi_nn_ConvertTensorDtype( self->graph, t1, &dst_dtype );
                vsi_safe_release_tensor(t1);
            }
            else
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_LN_WI + i] = t1;
            }

            inputs[LSTMUNIT_ACT_LN_WI + i] = p->local.tensors[LSTMUNIT_ACT_TENSOR_LN_WI + i];
        }
    }

    if( VSI_NN_DIM_AUTO == outputs[LSTMUNIT_ACT_OUTPUT]->attr.dim_num )
    {
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.dim_num = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.dim_num;
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[0] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[0];
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[1] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[1];
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[2] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[2];
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[3] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[3];
    }

    if( VSI_NN_DIM_AUTO == outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.dim_num )
    {
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.dim_num = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.dim_num;
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[0] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[0];
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[1] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[1];
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[2] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[2];
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[3] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[3];
    }

    if (outputs[LSTMUNIT_ACT_HSTATE_OUT] && VSI_NN_DIM_AUTO == outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.dim_num )
    {
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.dim_num = outputs[LSTMUNIT_ACT_OUTPUT]->attr.dim_num;
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[0] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[0];
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[1] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[1];
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[2] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[2];
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[3] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[3];
    }

    ret = TRUE;
final:
    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    int32_t i = 0;

    for (i = 0; i < LSTMUNIT_ACT_TENSOR_CNT; i++)
    {
        if (self->nn_param.lstmunit_activation.local.tensors[i] != NULL)
        {
            vsi_nn_ReleaseTensor(&self->nn_param.lstmunit_activation.local.tensors[i]);
            self->nn_param.lstmunit_activation.local.tensors[i] = NULL;
        }
    }

    if(self->nn_param.lstmunit_activation.local.lstmunit_param != NULL)
    {
        vsi_nn_ReleaseTensor(&self->nn_param.lstmunit_activation.local.lstmunit_param);
        self->nn_param.lstmunit_activation.local.lstmunit_param = NULL;
    }

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.lstmunit_activation.recurrent_activation = VSI_NN_ACT_SIGMOID;

    return status;
} /* op_init() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTMUNIT_ACTIVATION,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cpluplus
}
#endif

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
#include "vsi_nn_error.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_rnn_helper.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
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

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t output_size = 0;
    vsi_size_t batch_size = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    batch_size = inputs[RNNCELL_INPUT_INPUT]->attr.size[1];
    output_size = inputs[RNNCELL_INPUT_WEIGHT_I]->attr.size[1];

    /* create h_state input/output if app doesn't provide them */
    if( !inputs[RNNCELL_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy( &attr.dtype, &outputs[RNNCELL_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[RNNCELL_INPUT_H_STATE] = output_tensor->t;
    }

    if( !outputs[RNNCELL_OUTPUT_H_STATE] )
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[RNNCELL_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        outputs[RNNCELL_OUTPUT_H_STATE] = output_tensor->t;
    }

    /* setup rnncell output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[RNNCELL_OUTPUT_OUTPUT]->attr.dim_num)
    {
        /* num_units */
        outputs[RNNCELL_OUTPUT_OUTPUT]->attr.size[0] = inputs[RNNCELL_INPUT_WEIGHT_I]->attr.size[1];
        /* batch_size */
        outputs[RNNCELL_OUTPUT_OUTPUT]->attr.size[1] = inputs[RNNCELL_INPUT_INPUT]->attr.size[1];
        outputs[RNNCELL_OUTPUT_OUTPUT]->attr.dim_num = inputs[RNNCELL_INPUT_INPUT]->attr.dim_num;
    }

    /* output_state_out */
    if(VSI_NN_DIM_AUTO == outputs[RNNCELL_OUTPUT_H_STATE]->attr.dim_num)
    {
        outputs[RNNCELL_OUTPUT_H_STATE]->attr.dim_num =
            outputs[RNNCELL_OUTPUT_OUTPUT]->attr.dim_num;
        memcpy( outputs[RNNCELL_OUTPUT_H_STATE]->attr.size,
            outputs[RNNCELL_OUTPUT_OUTPUT]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t) );
    }
    return TRUE;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_rnncell_ovxlib_param* p = &self->nn_param.rnncell_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_bool is_input_fc_on_tp = FALSE;
    vsi_bool is_hstate_fc_on_tp = FALSE;
    vsi_nn_internal_tensor_t* input_tensor = NULL;
    vsi_nn_internal_tensor_t* input_gate_fc_outputs = NULL;
    vsi_nn_internal_tensor_t* hstate_gate_fc_outputs = NULL;
    vsi_nn_internal_tensor_t* aux_input_gate_fc_outputs = NULL;
    vsi_nn_internal_tensor_t* input_add_hstate_outputs = NULL;
    vsi_nn_internal_tensor_t* gate_fc_outputs = NULL;
    vsi_nn_internal_tensor_t* hstate_input_tensor = NULL;
    vsi_nn_internal_tensor_t* tmp = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_bool use_virtual_tensor = TRUE;
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_node_wksp( self );
    p->local = (vsi_nn_rnncell_ovxlib_lcl_data_t*)
        malloc(sizeof(vsi_nn_rnncell_ovxlib_lcl_data_t));
    CHECK_PTR_FAIL_GOTO( p->local, "Create buffer fail.", final );
    ret = TRUE;

    memset(p->local, 0x00, sizeof(vsi_nn_rnncell_ovxlib_lcl_data_t));
    memset(&attr, 0x00, sizeof(attr));
    p->local->multi_batch = (vsi_bool)(inputs[RNNCELL_INPUT_INPUT]->attr.size[1]);

    if( inputs[RNNCELL_INPUT_INPUT]->attr.dtype.qnt_type
        != inputs[RNNCELL_INPUT_WEIGHT_I]->attr.dtype.qnt_type)
    {
        /* input and input weights have different qtype, only TP can do this operation */
        is_input_fc_on_tp = TRUE;
    }
    else if( inputs[RNNCELL_INPUT_INPUT]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if input's shape is not aligned to 64-byte */
        is_input_fc_on_tp = TRUE;
    }

    if( inputs[RNNCELL_INPUT_H_STATE]->attr.dtype.qnt_type
        != inputs[RNNCELL_INPUT_WEIGHT_H]->attr.dtype.qnt_type)
    {
        /* recurrent and recurrent weights have different qtype, only TP can do this operation */
        is_hstate_fc_on_tp = TRUE;
    }
    else if( inputs[RNNCELL_INPUT_H_STATE]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if inputs' shape is not aligned to 64-byte */
        is_hstate_fc_on_tp = TRUE;
    }

    /* if both input fc and recurrent fc could be executed on NN, offloads one to TP*/
    if( !is_input_fc_on_tp && !is_hstate_fc_on_tp )
    {
        is_input_fc_on_tp = TRUE;
    }
    /* TODO: now, all fc on tp because can't fetch the HW feature */
    is_input_fc_on_tp = TRUE;
    is_hstate_fc_on_tp = TRUE;

    setup_op_shapes(self, inputs, outputs);

    /* Input FC */
    if( is_input_fc_on_tp )
    {
        /* tp */
        input_gate_fc_outputs = vsi_nn_rnn_create_tp_fc(self,
                                    inputs[RNNCELL_INPUT_INPUT],
                                    inputs[RNNCELL_INPUT_WEIGHT_I],
                                    inputs[RNNCELL_INPUT_BIAS],
                                    &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_I],
                                    use_virtual_tensor);
        if (inputs[RNNCELL_INPUT_AUX_INPUT] != NULL)
        {
            aux_input_gate_fc_outputs = vsi_nn_rnn_create_tp_fc(self,
                                            inputs[RNNCELL_INPUT_AUX_INPUT],
                                            inputs[RNNCELL_INPUT_AUX_WEIGHT],
                                            NULL,
                                            &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_AUX],
                                            use_virtual_tensor);
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            (uint32_t)inputs[RNNCELL_INPUT_INPUT]->attr.size[0],
            &kernel_h, &kernel_w);
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[RNNCELL_INPUT_INPUT],
            p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        tmp = vsi_nn_rnn_create_nn_fc(self,
                input_tensor->t,
                inputs[RNNCELL_INPUT_WEIGHT_I],
                inputs[RNNCELL_INPUT_BIAS],
                kernel_h, kernel_w,
                &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_I],
                use_virtual_tensor);
        /* transpose and reshape output */
        input_gate_fc_outputs = vsi_nn_rnn_process_output_for_nn_fc(self, tmp->t, p->local->multi_batch, kernel_h,
            kernel_w, use_virtual_tensor);
        if (inputs[RNNCELL_INPUT_AUX_INPUT] != NULL)
        {
            /* reshape and transpose input */
            vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
                (uint32_t)inputs[RNNCELL_INPUT_AUX_INPUT]->attr.size[0],
                &kernel_h, &kernel_w);
            input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self,
                            inputs[RNNCELL_INPUT_AUX_INPUT],
                            p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
            tmp = vsi_nn_rnn_create_nn_fc(self,
                    input_tensor->t,
                    inputs[RNNCELL_INPUT_AUX_INPUT],
                    NULL,
                    kernel_h, kernel_w,
                    &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_AUX],
                    use_virtual_tensor);
            /* transpose and reshape output */
            aux_input_gate_fc_outputs = vsi_nn_rnn_process_output_for_nn_fc(self,
                                            tmp->t, p->local->multi_batch, kernel_h,
                                            kernel_w, use_virtual_tensor);
        }
    }

    /* Hstate FC */
    if( is_hstate_fc_on_tp )
    {
        hstate_gate_fc_outputs = vsi_nn_rnn_create_tp_fc(self,
                                    inputs[RNNCELL_INPUT_H_STATE],
                                    inputs[RNNCELL_INPUT_WEIGHT_H],
                                    NULL,
                                    &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_H],
                                    use_virtual_tensor);
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            (uint32_t)inputs[RNNCELL_INPUT_H_STATE]->attr.size[0], &kernel_h, &kernel_w);
        hstate_input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self,
                                inputs[RNNCELL_INPUT_H_STATE],
                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        tmp = vsi_nn_rnn_create_nn_fc(self,
                hstate_input_tensor->t,
                inputs[RNNCELL_INPUT_WEIGHT_H],
                NULL,
                kernel_h, kernel_w,
                &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_H],
                use_virtual_tensor);
        /* transpose and reshape output */
        hstate_gate_fc_outputs = vsi_nn_rnn_process_output_for_nn_fc(self,
            tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
    }

    input_add_hstate_outputs = vsi_nn_rnn_create_tensor_add(self,
                                    input_gate_fc_outputs->t,
                                    hstate_gate_fc_outputs->t,
                                    &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_I],
                                    use_virtual_tensor);

    if (inputs[RNNCELL_INPUT_AUX_INPUT] != NULL)
    {
        gate_fc_outputs = vsi_nn_rnn_create_tensor_add(self,
                            input_add_hstate_outputs->t,
                            aux_input_gate_fc_outputs->t,
                            &p->internal_dtype[RNNCELL_QUANTIZE_PARAM_I],
                            use_virtual_tensor);
    }
    else
    {
        gate_fc_outputs = input_add_hstate_outputs;
    }

    /* activation */
    curr = vsi_nn_internal_new_node( self, vsi_nn_rnn_get_act_op_type(p->activation), 0, 0 );
    curr->inputs[0] = gate_fc_outputs->t;
    curr->outputs[0] = outputs[RNNCELL_OUTPUT_OUTPUT];
    vsi_nn_internal_setup_node(self, curr);

    if (outputs[RNNCELL_OUTPUT_H_STATE] != NULL)
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0] = outputs[RNNCELL_OUTPUT_OUTPUT];
        curr->outputs[0] = outputs[RNNCELL_OUTPUT_H_STATE];
        vsi_nn_internal_setup_node(self, curr);
    }

final:
    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_rnncell_ovxlib_param* p = &self->nn_param.rnncell_ovxlib;
    vsi_nn_safe_free(p->local);
    vsi_nn_safe_free(p->internal_dtype);
    vsi_nn_internal_deinit_node_wksp( self );

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_FAILURE;

    self->nn_param.rnncell_ovxlib.local = (vsi_nn_rnncell_ovxlib_lcl_data_t *)
        malloc(sizeof(vsi_nn_rnncell_ovxlib_lcl_data_t));
    CHECK_PTR_FAIL_GOTO( self->nn_param.rnncell_ovxlib.local, "Create buffer fail.", final );
    memset(self->nn_param.rnncell_ovxlib.local, 0,
        sizeof(vsi_nn_rnncell_ovxlib_lcl_data_t));
    self->nn_param.rnncell_ovxlib.internal_dtype = (vsi_nn_dtype_t *)
        malloc(sizeof(vsi_nn_dtype_t) * RNNCELL_QUANTIZE_PARAM_COUNT);
    CHECK_PTR_FAIL_GOTO( self->nn_param.rnncell_ovxlib.internal_dtype, "Create buffer fail.", final );
    memset(self->nn_param.rnncell_ovxlib.internal_dtype, 0,
        sizeof(vsi_nn_dtype_t) * RNNCELL_QUANTIZE_PARAM_COUNT);

    status = VSI_SUCCESS;
final:
    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RNNCELL_OVXLIB,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ RNNCELL_INPUT_CNT,
    /* output_num */ RNNCELL_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif

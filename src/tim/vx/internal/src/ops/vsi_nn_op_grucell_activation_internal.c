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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_activation_internal_param* p = &self->nn_param.grucell_activation_internal;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t* param;

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32(param, "gate_activation", p->gate_activation);
    vsi_nn_kernel_param_add_int32(param, "candidate_activation", p->candidate_activation);
    vsi_nn_kernel_param_add_int32(param, "input_category", p->input_category);
    vsi_nn_kernel_param_add_int32(param, "use_cudnn_implementation", p->use_cudnn_implementation);
    vsi_nn_kernel_param_add_int32(param, "input_layout", p->input_layout);

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "grucell_activation",
        inputs, GRUCELL_ACTIVATION_INPUT_COUNT,
        outputs, GRUCELL_ACTIVATION_OUTPUT_COUNT,
        param );

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
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /**
     * input 0, input 1, input2
     * z{t_}, h{t__}, h{t-1}
     * z{t} = gate_activation(z{t_})
     * h{t_} = candidate_activation(h{t__})
     * h{t} = z{t} * (h{t-1} - h{t_}) + h{t_}
     */
    vsi_nn_grucell_activation_internal_param* p = &self->nn_param.grucell_activation_internal;

    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.dim_num)
    {
        if(p->input_category == GRUCELL_INPUT_CATEGORY_DEFAULT)
        {
            outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.dim_num = \
                inputs[GRUCELL_ACTIVATION_INPUT_ZT_]->attr.dim_num;
            memcpy( outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size,
                inputs[GRUCELL_ACTIVATION_INPUT_ZT_]->attr.size,
                inputs[GRUCELL_ACTIVATION_INPUT_ZT_]->attr.dim_num * sizeof(vsi_size_t) );
        }
        else
        {
            outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.dim_num = \
                inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr.dim_num;

            if(GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC == p->input_layout)
            {
                outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size[1] = \
                    inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr.size[1];
                if(inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z])
                {
                    outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size[0] = \
                        inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z]->attr.size[0];
                }
                else
                {
                    /* for batch first, the inputs of Z/R/C are always concaetnated in axis-0 */
                    outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size[0] = \
                        inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr.size[0] / 3;
                }
            }
            else
            {
                outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size[0] = \
                    inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr.size[0];

                if(inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z])
                {
                    outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size[1] = \
                        inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z]->attr.size[1];
                }
                /* if `Z` is not provided, the inputs of Z/R/C are concatenated in axis-1 */
                else
                {
                    outputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size[1] = \
                        inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr.size[1] / 3;
                }
            }
        }
    }
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_ACTIVATION_OUTPUT_H_STATE]->attr.dim_num)
    {
        outputs[GRUCELL_ACTIVATION_OUTPUT_H_STATE]->attr.dim_num = \
            inputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.dim_num;
        memcpy( outputs[GRUCELL_ACTIVATION_OUTPUT_H_STATE]->attr.size,
            inputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.size,
            inputs[GRUCELL_ACTIVATION_OUTPUT_OUTPUT]->attr.dim_num * sizeof(vsi_size_t) );
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.grucell_activation_internal.local = \
        (vsi_nn_grucell_activation_internal_local *)malloc(sizeof(vsi_nn_grucell_activation_internal_local));
    self->nn_param.grucell_activation_internal.gate_activation = VSI_NN_ACT_SIGMOID;
    self->nn_param.grucell_activation_internal.candidate_activation = VSI_NN_ACT_TANH;
    self->nn_param.grucell_activation_internal.input_category = GRUCELL_INPUT_CATEGORY_DEFAULT;
    self->nn_param.grucell_activation_internal.use_cudnn_implementation = FALSE;
    self->nn_param.grucell_activation_internal.input_layout = GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_safe_free(self->nn_param.grucell_activation_internal.local);
    status = vsi_nn_op_common_deinit(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRUCELL_ACTIVATION_INTERNAL,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ GRUCELL_ACTIVATION_INPUT_COUNT,
    /* output_num */ GRUCELL_ACTIVATION_OUTPUT_COUNT
    );

__END_DECLS

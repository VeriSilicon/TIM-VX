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


/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;

    vsi_nn_kernel_param_t* param = NULL;
    int32_t align_corners = self->nn_param.gridsample.align_corners;
    vsi_nn_kernel_node_t n;
    char kernel_name[128];

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32(param, "align_corners", align_corners);

    switch (self->nn_param.gridsample.mode) {
        case VSI_NN_INTERPOLATION_BILINEAR:
            snprintf(kernel_name, sizeof(kernel_name), "bilinear_grid_sample");
            break;
        case VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR:
            snprintf(kernel_name, sizeof(kernel_name), "nearest_grid_sample");
            break;
        default:
            break;
    }

    n = (vx_node)vsi_nn_kernel_selector(
        self->graph, kernel_name, inputs, 2, outputs, 1, param);

    if (n == NULL) {
        vsi_nn_kernel_param_release(&param);
        status = VSI_FAILURE;
        return status;
    }
    self->n = (vx_node)n;
    vsi_nn_kernel_param_release(&param);
    if (self->n) {
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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    if ((VSI_NN_INTERPOLATION_BILINEAR != self->nn_param.gridsample.mode) &&
        (VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR !=
         self->nn_param.gridsample.mode)) {
        VSILOGE("Only support bilinear or nearest grid sample mode now!");
        return FALSE;
    }

    if (!((VSI_NN_PAD_MODE_CONSTANT ==
           self->nn_param.gridsample.padding_mode) &&
          (0 == self->nn_param.gridsample.const_val))) {
        VSILOGE("Only support padding const 0 now!");
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
    if (NULL == self) {
        return FALSE;
    }

    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num) {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = inputs[1]->attr.size[1];
        outputs[0]->attr.size[1] = inputs[1]->attr.size[2];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        if (4 == inputs[0]->attr.dim_num) {
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        }
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    /* TODO
    //self->nn_param.grid_sample.local = \
    //    (grid_sample_local_data_t*)malloc(sizeof(grid_sample_local_data_t));
    */
    VSI_UNREFERENCED(self);
    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_op_common_deinit(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRID_SAMPLE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS


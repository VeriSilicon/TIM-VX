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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_error.h"

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
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_kernel_node_t    n = NULL;
    vsi_nn_tensor_t* reshape_tensor = NULL;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_pre_process_rgb_param* p = NULL;

    memcpy(shape, inputs[0]->attr.size, inputs[0]->attr.dim_num * sizeof(vsi_size_t));

    shape[0] = shape[1] * shape[0];
    shape[1] = shape[2];
    shape[2] = 1;

    reshape_tensor = vsi_nn_reshape_tensor(self->graph,
        inputs[0], shape, inputs[0]->attr.dim_num);
    CHECK_PTR_FAIL_GOTO(reshape_tensor, "Create tensor failed", final);

    p = (vsi_nn_pre_process_rgb_param*)&(self->nn_param.pre_process_rgb);

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "scale_x", p->local->scale_x );
    vsi_nn_kernel_param_add_int32( param, "scale_y", p->local->scale_y );
    vsi_nn_kernel_param_add_int32( param, "left", p->rect.left );
    vsi_nn_kernel_param_add_int32( param, "top", p->rect.top );
    vsi_nn_kernel_param_add_float32( param, "r_mean", p->r_mean );
    vsi_nn_kernel_param_add_float32( param, "g_mean", p->g_mean );
    vsi_nn_kernel_param_add_float32( param, "b_mean", p->b_mean );
    vsi_nn_kernel_param_add_float32( param, "r_scale", p->r_scale );
    vsi_nn_kernel_param_add_float32( param, "g_scale", p->g_scale );
    vsi_nn_kernel_param_add_float32( param, "b_scale", p->b_scale );
    vsi_nn_kernel_param_add_int32( param, "reverse", p->reverse_channel );
    vsi_nn_kernel_param_add_int32( param, "enable_perm", p->local->enable_perm );
    vsi_nn_kernel_param_add_int32( param, "enable_copy", p->local->enable_copy );
    n = vsi_nn_kernel_selector( self->graph, "pre_process_rgb", &reshape_tensor, 1, outputs, 1, param );
    if ( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
    }

final:
    vsi_safe_release_tensor(reshape_tensor);

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(PRE_PROCESS_RGB, 1, 1)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8,  D_U8|Q_ASYM)
        IO_TYPE(D_U8,  D_I8|Q_DFP)
        IO_TYPE(D_U8,  D_I16|Q_DFP)
        IO_TYPE(D_U8,  D_F16)
    END_IO_TYPE_DECL(PRE_PROCESS_RGB)
    if(!VALIDATE_OP_IO_TYPES(PRE_PROCESS_RGB, self, inputs, self->input.num, outputs, self->output.num)) {
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
    vsi_nn_pre_process_rgb_param * p = NULL;
    uint32_t i = 0;

    VSI_UNREFERENCED(inputs);

    p = (vsi_nn_pre_process_rgb_param *)&(self->nn_param.pre_process_rgb);

    if (p->rect.width == 0 || p->rect.height == 0)
    {
        VSILOGE("Image size cannot be zero !(PRE_PROCESS_RGB)\n");
        return FALSE;
    }
    else
    {
        for (i = 0; i < p->output_attr.dim_num; i++)
        {
            if (p->output_attr.size[i] == 0)
            {
                VSILOGE("output size cannot be zero!(PRE_PROCESS_RGB)\n");
                return FALSE;
            }
        }
    }

    if( VSI_NN_DIM_AUTO == outputs[PRE_PROCESS_RGB_OUTPUT]->attr.dim_num )
    {
        if (p->output_attr.dim_num > 0)
        {
            for (i = 0; i < p->output_attr.dim_num; i++)
            {
                if (p->output_attr.size[i] == 0)
                {
                    VSILOGE("output size cannot be zero!(PRE_PROCESS_RGB)\n");
                    return FALSE;
                }
                else
                {
                    outputs[PRE_PROCESS_RGB_OUTPUT]->attr.dim_num = p->output_attr.dim_num;
                    outputs[PRE_PROCESS_RGB_OUTPUT]->attr.size[i] = p->output_attr.size[i];
                }
            }
        }
        else
        {
            VSILOGE("output dim num cannot be zero!(PRE_PROCESS_RGB)\n");
            return FALSE;
        }
    }


    p->local->enable_perm = FALSE;

    if (p->local->enable_perm == FALSE)
    {
        p->local->scale_x = (int32_t)((p->rect.width << 15) / outputs[PRE_PROCESS_RGB_OUTPUT]->attr.size[0]);
        p->local->scale_y = (int32_t)((p->rect.height << 15) / outputs[PRE_PROCESS_RGB_OUTPUT]->attr.size[1]);
    }
    else
    {
        p->local->scale_x = (int32_t)((p->rect.width << 15) / outputs[PRE_PROCESS_RGB_OUTPUT]->attr.size[1]);
        p->local->scale_y = (int32_t)((p->rect.height << 15) / outputs[PRE_PROCESS_RGB_OUTPUT]->attr.size[2]);
    }

    p->local->enable_copy = ((p->local->scale_x == p->local->scale_y) && (p->local->scale_x == (1 << 15)));

    return TRUE;
} /* op_setup() */

static vsi_status op_init
(
    vsi_nn_node_t* self
)
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.pre_process_rgb.local =
        (vsi_nn_pre_process_rgb_lcl_data*)malloc(sizeof(vsi_nn_pre_process_rgb_lcl_data));

    if (NULL == self->nn_param.pre_process_rgb.local)
    {
        return VX_ERROR_NO_MEMORY;
    }

    memset(self->nn_param.pre_process_rgb.local, 0, sizeof(vsi_nn_pre_process_rgb_lcl_data));

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.pre_process_rgb.local->local_tensor != NULL)
    {
        vxReleaseTensor(&self->nn_param.pre_process_rgb.local->local_tensor);
        self->nn_param.pre_process_rgb.local->local_tensor = NULL;
    }

    vsi_nn_safe_free(self->nn_param.pre_process_rgb.local);

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PRE_PROCESS_RGB,
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

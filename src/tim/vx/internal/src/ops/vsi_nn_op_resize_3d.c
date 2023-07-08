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
#include "vsi_nn_error.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_tensor_util.h"

typedef struct _resize_3d_local_data_t {
    int32_t placeholder;
} resize_3d_local_data_t;

/*
 Declare number of input and output.
 */
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
    vsi_nn_tensor_t *      reshape_inputs[1]     = {NULL};
    vsi_nn_tensor_t *      reshape_outputs[1]    = {NULL};

    if ( self->nn_param.resize_3d.lcl_data->use_internal_node )
    {
        status = vsi_nn_internal_compute_node( self );
    }
    else
    {
        char kernel_name[128];
        vsi_nn_kernel_param_t * param = NULL;
        int32_t align_corners = self->nn_param.resize_3d.align_corners;
        int32_t half_pixel_centers = self->nn_param.resize_3d.half_pixel_centers;
        vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM]     = {{0}};
        uint32_t               new_rank              = 4;
        uint32_t i = 0;

        if (inputs[0]->attr.dim_num > 3)
        {
            shapes[0][0] = inputs[0]->attr.size[0];
            shapes[0][1] = inputs[0]->attr.size[1];
            shapes[0][2] = inputs[0]->attr.size[2];
            shapes[1][0] = outputs[0]->attr.size[0];
            shapes[1][1] = outputs[0]->attr.size[1];
            shapes[1][2] = outputs[0]->attr.size[2];
            shapes[0][3] = 1;
            shapes[1][3] = 1;

            for (i = 3; i < inputs[0]->attr.dim_num; i++)
            {
                shapes[0][3] = shapes[0][3] * inputs[0]->attr.size[i];
            }
            shapes[1][3] = shapes[0][3];

            reshape_inputs[0] = vsi_nn_reshape_tensor(self->graph, inputs[0], shapes[0], new_rank);
            reshape_outputs[0] = vsi_nn_reshape_tensor(self->graph, outputs[0], shapes[1], new_rank);

            if (reshape_inputs[0] == NULL || reshape_outputs[0] == NULL)
            {
                VSILOGE("reshape tensor failed");
                status = VSI_FAILURE;
                goto final;
            }
        }
        else
        {
            reshape_inputs[0] = inputs[0];
            reshape_outputs[0] = outputs[0];
        }


        param = vsi_nn_kernel_param_create();

        vsi_nn_kernel_param_add_int32( param, "align_corners",  align_corners );
        vsi_nn_kernel_param_add_int32( param, "half_pixel_centers",  half_pixel_centers );
        vsi_nn_kernel_param_add_int32( param, "type",  self->nn_param.resize_3d.type );

        switch (self->nn_param.resize_3d.type)
        {
            case VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR:
                 snprintf(kernel_name, sizeof(kernel_name),
                 "resize_3d_nearest");
                 break;
            case VSI_NN_INTERPOLATION_BILINEAR:
                 snprintf(kernel_name, sizeof(kernel_name),
                 "resize_3d_bilinear");
                 break;
            default:
                break;
        }

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
            kernel_name, &reshape_inputs[0], 1, &reshape_outputs[0], 1, param );

        if (self->n) {
            status = VSI_SUCCESS;
        }

        vsi_nn_kernel_param_release(&param);
    }

final:
    vsi_safe_release_tensor( reshape_inputs[0] );
    vsi_safe_release_tensor( reshape_outputs[0] );

    return status;
} /* op_compute() */

static vsi_bool _is_same_shape
    (
    vsi_nn_tensor_t * inputs,
    vsi_size_t *sizes,
    uint32_t dims
    )
{
    uint32_t i = 0;

    if (inputs->attr.dim_num != dims)
        return FALSE;

    for (i = 0; i < dims; i++)
    {
        if (sizes[i] != inputs->attr.size[i])
            return FALSE;
    }

    return TRUE;
}

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if ( self->nn_param.resize_3d.lcl_data->use_internal_node )
    {
        return vsi_nn_internal_optimize_node(self, direction );
    }
    else
    {
        int32_t half_pixel_centers = self->nn_param.resize_3d.half_pixel_centers;
        vsi_size_t * input_size = inputs[0]->attr.size;
        vsi_size_t * output_size = outputs[0]->attr.size;

        if ( (output_size[0] % input_size[0] == 0) && (output_size[1] % input_size[1] == 0) &&
            half_pixel_centers == TRUE && self->nn_param.resize_3d.type == VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR )
        {
            self->nn_param.resize_3d.half_pixel_centers = FALSE;
        }

        return VSI_SUCCESS;
    }
} /* op_optimize() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(RESIZE_3D, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,  D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM, D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_I16|Q_SYM)
    END_IO_TYPE_DECL(RESIZE_3D)
    if (!VALIDATE_OP_IO_TYPES(RESIZE_3D, self, inputs, self->input.num, outputs, self->output.num)) {
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
    float factor = self->nn_param.resize_3d.factor;
    vsi_nn_internal_node_t* curr = NULL;
    uint32_t i = 0;
    vsi_bool ret = TRUE;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        if (factor != 0)
        {
            outputs[0]->attr.size[0] = (uint32_t)(inputs[0]->attr.size[0] * factor);
            outputs[0]->attr.size[1] = (uint32_t)(inputs[0]->attr.size[1] * factor);
            outputs[0]->attr.size[2] = (uint32_t)(inputs[0]->attr.size[2] * factor);
        }
        else
        {
            outputs[0]->attr.size[0] = self->nn_param.resize_3d.size[0];
            outputs[0]->attr.size[1] = self->nn_param.resize_3d.size[1];
            outputs[0]->attr.size[2] = self->nn_param.resize_3d.size[2];
        }
        for (i = 3; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
    }

    if (_is_same_shape(inputs[0], outputs[0]->attr.size, outputs[0]->attr.dim_num))
    {
        self->nn_param.resize.lcl_data->use_internal_node = TRUE;
        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        ret = vsi_nn_internal_setup_node(self, curr);
    }

final:
    return ret;
} /* op_setup() */

static vsi_status op_init(vsi_nn_node_t* self) {
    vsi_status status = VSI_SUCCESS;

    self->nn_param.resize_3d.lcl_data =
        (vsi_nn_resize_3d_local_data*)malloc(sizeof(vsi_nn_resize_3d_local_data));
    if (NULL == self->nn_param.resize_3d.lcl_data) {
        VSILOGE("Create resize_3d local data fail.");
        status = VSI_FAILURE;
        goto final;
    }
    memset(self->nn_param.resize_3d.lcl_data, 0, sizeof(vsi_nn_resize_3d_local_data));

    self->nn_param.resize_3d.align_corners = FALSE;
    self->nn_param.resize_3d.half_pixel_centers = FALSE;


final:
    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    if (self->nn_param.resize_3d.lcl_data->use_internal_node)
    {
        vsi_nn_safe_free(self->nn_param.resize_3d.lcl_data);
        vsi_nn_internal_deinit_node_wksp(self);
    }
    else
    {
        vsi_nn_safe_free(self->nn_param.resize_3d.lcl_data);
        vsi_nn_op_common_deinit(self);
    }

    return VSI_SUCCESS;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RESIZE_3D,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS


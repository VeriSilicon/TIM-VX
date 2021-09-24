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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)


static vsi_status _reduce_internal_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    int32_t axis = 0;
    int32_t new_axis = 0;
    uint32_t axis_size = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;



    param =vsi_nn_kernel_param_create();

    if (strcmp(kernel_name, "reducemax_internal") == 0)
    {
        vsi_nn_reducemax_internal_param * p = &(self->nn_param.reducemax_internal);
        axis = p->axis[0];
    }
    else if (strcmp(kernel_name, "reducemin_internal") == 0)
    {
        vsi_nn_reducemin_internal_param * p = &(self->nn_param.reducemin_internal);
        axis = p->axis[0];
    }
    else if (strcmp(kernel_name, "reduceprod_internal") == 0)
    {
        vsi_nn_reduceprod_internal_param * p = &(self->nn_param.reduceprod_internal);
        axis = p->axis[0];
    }
    else if (strcmp(kernel_name, "reduceall_internal") == 0)
    {
        vsi_nn_reduceall_internal_param * p = &(self->nn_param.reduceall_internal);
        axis = p->axis[0];
    }
    else if (strcmp(kernel_name, "reduceany_internal") == 0)
    {
        vsi_nn_reduceany_internal_param * p = &(self->nn_param.reduceany_internal);
        axis = p->axis[0];
    }
    else
    {
        vsi_nn_kernel_param_release( &param );
        return VSI_FAILURE;

    }

    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            &axis, 1,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            &new_axis, &axis_size);

    // Add params
     vsi_nn_kernel_param_add_int32( param, "axis", new_axis );

    if( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[1], rank_out );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                kernel_name,
                &reshape_tensors[0], 1,
                &reshape_tensors[1], 1, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );
    return status;
} /* op_compute() */

static vsi_bool _reduce_internal_op_setup
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t axis = 0;

    if (strcmp(kernel_name, "reducemax_internal") == 0)
    {
        vsi_nn_reducemax_internal_param * p = &(self->nn_param.reducemax_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    else if (strcmp(kernel_name, "reducemin_internal") == 0)
    {
        vsi_nn_reducemin_internal_param * p = &(self->nn_param.reducemin_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    else if (strcmp(kernel_name, "reduceprod_internal") == 0)
    {
        vsi_nn_reduceprod_internal_param * p = &(self->nn_param.reduceprod_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    else if (strcmp(kernel_name, "reduceall_internal") == 0)
    {
        vsi_nn_reduceall_internal_param * p = &(self->nn_param.reduceall_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    else if (strcmp(kernel_name, "reduceany_internal") == 0)
    {
        vsi_nn_reduceany_internal_param * p = &(self->nn_param.reduceany_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    else
    {
         return FALSE;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t i = 0;
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num - 1;

        for (i = 0; i < (uint32_t)axis; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }

        for (i = axis; i < outputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i + 1];
        }

        if (inputs[0]->attr.dim_num == 1)
        {
            outputs[0]->attr.dim_num = 1;
            outputs[0]->attr.size[0] = 1;
        }
    }

    return TRUE;
} /* op_setup() */


static vsi_bool op_check_reduceall_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REDUCEALL_INTERNAL, 1, 1)
        IO_TYPE(D_I8, D_I8)
        IO_TYPE(D_BOOL8, D_BOOL8)
    END_IO_TYPE_DECL(REDUCEALL_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(REDUCEALL_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */


static vsi_bool op_check_reduceany_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REDUCEANY_INTERNAL, 1, 1)
        IO_TYPE(D_I8, D_I8)
        IO_TYPE(D_BOOL8, D_BOOL8)
    END_IO_TYPE_DECL(REDUCEANY_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(REDUCEANY_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */


static vsi_bool op_check_reducemax_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REDUCEMAX_INTERNAL, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_I32,  D_I16)
        IO_TYPE(D_I16,  D_I32)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
    END_IO_TYPE_DECL(REDUCEMAX_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(REDUCEMAX_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_check_reducemin_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REDUCEMIN_INTERNAL, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
    END_IO_TYPE_DECL(REDUCEMIN_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(REDUCEMIN_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_check_reduceprod_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REDUCEPROD_INTERNAL, 1, 1)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
    END_IO_TYPE_DECL(REDUCEPROD_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(REDUCEPROD_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_REDUCE_INTERNAL_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _reduce_internal_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
            static vsi_bool op_setup_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _reduce_internal_op_setup( ""#kernel_name, self, inputs, outputs ); \
            } \
DEF_OP_REG  \
    ( \
    /* op_name    */ name, \
    /* init       */ NULL, \
    /* compute    */ op_compute_##kernel_name, \
    /* deinit     */ vsi_nn_op_common_deinit, \
    /* check      */ op_check_##kernel_name, \
    /* setup      */ op_setup_##kernel_name, \
    /* optimize   */ NULL, \
    /* input_num  */ 1, \
    /* output_num */ 1 \
    )


DEF_REDUCE_INTERNAL_OP( REDUCEMAX_INTERNAL,  reducemax_internal );
DEF_REDUCE_INTERNAL_OP( REDUCEMIN_INTERNAL,  reducemin_internal );
DEF_REDUCE_INTERNAL_OP( REDUCEPROD_INTERNAL, reduceprod_internal );
DEF_REDUCE_INTERNAL_OP( REDUCEALL_INTERNAL,  reduceall_internal );
DEF_REDUCE_INTERNAL_OP( REDUCEANY_INTERNAL,  reduceany_internal );

#undef DEF_REDUCE_INTERNAL_OP
#ifdef __cplusplus
}
#endif

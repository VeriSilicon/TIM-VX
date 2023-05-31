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
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _conv3d_local_data_t {
    int32_t placeholder;
} conv3d_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (3)
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
    param = vsi_nn_kernel_param_create();

#define MAP_PARAM(type_name, value) {\
    vsi_nn_kernel_param_add_int32( param, type_name, value); \
    }

    MAP_PARAM("stride_w",self->nn_param.conv3d.stride[0]);
    MAP_PARAM("stride_h",self->nn_param.conv3d.stride[1]);
    MAP_PARAM("stride_d",self->nn_param.conv3d.stride[2]);

    MAP_PARAM("pad_left",self->nn_param.conv3d.pad[0]);
    MAP_PARAM("pad_right",self->nn_param.conv3d.pad[1]);
    MAP_PARAM("pad_top",self->nn_param.conv3d.pad[2]);
    MAP_PARAM("pad_bottom",self->nn_param.conv3d.pad[3]);
    MAP_PARAM("pad_front",self->nn_param.conv3d.pad[4]);
    MAP_PARAM("pad_end",self->nn_param.conv3d.pad[5]);

    MAP_PARAM("depth_multiplier", self->nn_param.conv3d.multiplier);
    MAP_PARAM("overflow_policy",self->vx_param.overflow_policy);
    MAP_PARAM("rounding_policy",self->vx_param.rounding_policy);
    MAP_PARAM("down_scale_size_rounding",self->vx_param.down_scale_size_rounding);
    MAP_PARAM("pad_mode", vsi_nn_get_vx_pad_mode( self->nn_param.conv3d.pad_mode ) );

    if ( self->nn_param.conv3d.dilation[0] *
         self->nn_param.conv3d.dilation[1] *
         self->nn_param.conv3d.dilation[2] > 1)
    {
        VSILOGE("conv3d could not support dilation > 1\n");
        goto final;
    }else
    {
        MAP_PARAM("dilation_w",self->nn_param.conv3d.dilation[0]);
        MAP_PARAM("dilation_h",self->nn_param.conv3d.dilation[1]);
        MAP_PARAM("dilation_d",self->nn_param.conv3d.dilation[2]);
    }
#undef MAP_PARAM
    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "conv3d",
            inputs, 3, outputs, 1, param );
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

final:
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
    vsi_bool ret = FALSE;

    /* Check fl and scale*/
    ret = vsi_nn_QuantCheck(inputs[0], inputs[1], inputs[2]);

    if(ret) {
         /* check inputs outputs data type */
        BEGIN_IO_TYPE_DECL(CONV3D, 3, 1)

            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I32|Q_DFP, D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I32|Q_DFP, D_F16)

            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I32|Q_DFP, D_I16|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I64|Q_DFP, D_I16|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I64|Q_DFP, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP, D_I32|Q_ASYM, D_U8|Q_ASYM)

            IO_TYPE(D_BF16, D_BF16, D_F32, D_F32)
            IO_TYPE(D_BF16, D_BF16, D_F32, D_BF16)
            IO_TYPE(D_F16, D_F16, D_F32, D_F16)
            IO_TYPE(D_F16, D_F16, D_F32, D_F32)

            IO_TYPE(D_F32, D_F32, D_F32, D_F32)

            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_I8|Q_ASYM)
            IO_TYPE(D_I8|Q_SYM, D_I8|Q_SYM, D_I32|Q_SYM, D_I8|Q_SYM)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_I32|Q_SYM_PC, D_U8|Q_ASYM)

            /* IO_TYPE(INPUT, WEIGHT, NULL, OUTPUT) */
            IO_TYPE(D_F32, D_F32, D_NONE, D_F32)

            IO_TYPE(D_F16, D_F16, D_NONE, D_F16)

            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_NONE, D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_NONE, D_F16)

            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_NONE, D_I16|Q_DFP)

            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_NONE, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_NONE, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP, D_NONE, D_U8|Q_ASYM)

            IO_TYPE(D_BF16, D_BF16, D_NONE, D_F32)
            IO_TYPE(D_BF16, D_BF16, D_NONE, D_BF16)

            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_NONE, D_I8|Q_ASYM)
            IO_TYPE(D_I8|Q_SYM, D_I8|Q_SYM, D_NONE, D_I8|Q_SYM)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC, D_NONE, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_NONE, D_U8|Q_ASYM)

            /* HW 9.0 */
            IO_TYPE(D_F32, D_BF16, D_F32, D_BF16)
            IO_TYPE(D_F32, D_BF16, D_NONE, D_BF16)
            /* HW 9.0.1 */
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_F32)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)

            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_F32)

        END_IO_TYPE_DECL(CONV3D)
        ret = VALIDATE_OP_IO_TYPES(CONV3D, self, inputs, self->input.num, outputs, self->output.num);
        if(!ret) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
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
    vsi_nn_conv3d_param *nn_param;
    vsi_size_t i, pad[_cnt_of_array(self->nn_param.conv3d.pad)] = {0};
    for(i = 0; i < _cnt_of_array(self->nn_param.conv3d.pad); i++)
    {
        pad[i] = self->nn_param.conv3d.pad[i];
    }
#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    nn_param = &self->nn_param.conv3d;

    vsi_nn_compute_padding_3d(
        inputs[0]->attr.size,
        inputs[1]->attr.size,
        (uint32_t *)self->nn_param.conv3d.stride,
        (uint32_t *)self->nn_param.conv3d.dilation,
        self->nn_param.conv3d.pad_type,
        pad
    );
    for(i = 0; i < _cnt_of_array(self->nn_param.conv3d.pad); i++)
    {
        self->nn_param.conv3d.pad[i] = (uint32_t)pad[i];
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            inputs[1]->attr.size[0],
            (vx_uint32 *)&nn_param->pad[0],
            nn_param->stride[0],
            nn_param->dilation[0],
            VSI_NN_ROUND_FLOOR
            );
        outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[1],
            inputs[1]->attr.size[1],
            (vx_uint32 *)&nn_param->pad[2],
            nn_param->stride[1],
            nn_param->dilation[1],
            VSI_NN_ROUND_FLOOR
            );
        outputs[0]->attr.size[2] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[2],
            inputs[1]->attr.size[2],
            (vx_uint32 *)&nn_param->pad[4],
            nn_param->stride[2],
            nn_param->dilation[2],
            VSI_NN_ROUND_FLOOR
            );
        if(self->nn_param.conv3d.weights > 0)
        {
            outputs[0]->attr.size[3] = self->nn_param.conv3d.weights;
        }
        else if(self->nn_param.conv3d.multiplier > 0)
        {
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3] * self->nn_param.conv3d.multiplier;
        }
        else
        {
            outputs[0]->attr.size[3] = inputs[1]->attr.size[4];
        }
        outputs[0]->attr.size[4] = inputs[0]->attr.size[4];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    /* TODO
    //self->nn_param.conv3d.local = \
    //    (conv3d_local_data_t*)malloc(sizeof(conv3d_local_data_t));
    */

    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_op_common_deinit(self);

    /* TODO
    //vsi_nn_safe_free(self->nn_param.conv3d.local);
    */

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONV3D,
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

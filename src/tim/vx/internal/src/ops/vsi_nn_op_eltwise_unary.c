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
#include "utils/vsi_nn_constraint_check.h"

static vsi_status _eltwise_unary_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    float alpha = 0;
    float beta = 0;
    vsi_nn_kernel_param_t * param = NULL;

    if( NULL == self )
    {
        return status;
    }
    param = vsi_nn_kernel_param_create();

    if (strcmp(kernel_name, "elu") == 0)
    {
        alpha = self->nn_param.elu.alpha;
        beta = 1.0f;
    }
    else if (strcmp(kernel_name, "celu") == 0)
    {
        alpha = self->nn_param.celu.alpha;
    }
    else if (strcmp(kernel_name, "selu") == 0)
    {
        alpha = self->nn_param.selu.alpha;
        beta = self->nn_param.selu.gamma;
    }
    else
    {
        alpha = self->nn_param.hard_sigmoid.alpha;
        beta = self->nn_param.hard_sigmoid.beta;
    }
    vsi_nn_kernel_param_add_float32( param, "alpha", alpha );
    vsi_nn_kernel_param_add_float32( param, "beta", beta );

    // TODO: This optimzie is a hack for gpu path,
    // it should be moved to gpu kernel setup.
    if (strcmp(kernel_name, "gelu") == 0 && self->nn_param.gelu.approximate)
    {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                    "hard_gelu", inputs, 1, outputs, 1, param );
    }
    else if (strcmp(kernel_name, "elu") == 0 )
    {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                    "selu", inputs, 1, outputs, 1, param );
    }
    else
    {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
            kernel_name, inputs, 1, outputs, 1, param );
    }

    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* _eltwise_op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i, out_rank;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = TRUE;

    out_rank = inputs[0]->attr.dim_num;

    for (i = 0; i < out_rank; i++)
    {
        shape[i] = inputs[0]->attr.size[i];
    }
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = out_rank;
        memcpy( outputs[0]->attr.size, shape, out_rank * sizeof(vsi_size_t) );
    }
    else
    {
        vsi_size_t total_size_got;
        vsi_size_t total_size_expected;
        total_size_expected = vsi_nn_ShapeProduct( shape, out_rank );
        total_size_got = vsi_nn_ShapeProduct( outputs[0]->attr.size,
                outputs[0]->attr.dim_num );
        if ( total_size_expected != total_size_got )
        {
            VSILOGW("Output size mismatch, expect %"VSI_SIZE_T_SPECIFIER", but got %"VSI_SIZE_T_SPECIFIER"",
                    total_size_expected, total_size_got);
            ret = FALSE;
        }
    }

    return ret;
} /* op_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* check inputs outputs data type */
    BEGIN_IO_TYPE_DECL(ELTWISE_UNARY, 1, 1)
        /* IO_TYPE(INPUT, OUTPUT) */
        IO_TYPE(D_I32,          D_I32)
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_F32,          D_F16)
        IO_TYPE(D_F32,          D_BF16)
        IO_TYPE(D_F16,          D_F32)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_F16,          D_U8|Q_ASYM)
        IO_TYPE(D_F16,          D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_I8|Q_ASYM)
        IO_TYPE(D_F16,          D_I8|Q_SYM)
        IO_TYPE(D_F16,          D_I16|Q_DFP)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_BF16,         D_F32)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,    D_F16)
    END_IO_TYPE_DECL(ELTWISE_UNARY)
    if (!VALIDATE_OP_IO_TYPES(ELTWISE_UNARY, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status _eltwise_unary_op_init
    (
    const char * kernel_name,
    vsi_nn_node_t * self
    )
{
    if (vsi_nn_compareVersion(self->graph, 1, 1, 29) == -1 &&
        strcmp(kernel_name, "elu") == 0)
    {
        self->nn_param.elu.alpha = 1;
    }
    else if (strcmp(kernel_name, "hard_sigmoid") == 0)
    {
        self->nn_param.hard_sigmoid.alpha = 0.2f;
        self->nn_param.hard_sigmoid.beta = 0.5f;
    }
    else if (strcmp(kernel_name, "selu") == 0)
    {
        self->nn_param.selu.alpha = 1.67326319217681884765625f;
        self->nn_param.selu.gamma = 1.05070102214813232421875f;
    }

    return VSI_SUCCESS;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_ELEMENT_WISE_UNARY_OP(name, kernel_name) \
    static vsi_status op_compute_##kernel_name \
        ( \
        vsi_nn_node_t * self, \
        vsi_nn_tensor_t ** inputs, \
        vsi_nn_tensor_t ** outputs \
        ) \
    { \
        return _eltwise_unary_op_compute( ""#kernel_name, self, inputs, outputs ); \
    } \
    static vsi_status op_init_##kernel_name \
        ( \
        vsi_nn_node_t * self \
        ) \
    { \
        return _eltwise_unary_op_init( ""#kernel_name, self ); \
    } \
DEF_OP_REG(name, op_init_##kernel_name, op_compute_##kernel_name, \
    vsi_nn_op_common_deinit, op_check, op_setup, NULL, 1, 1)

DEF_ELEMENT_WISE_UNARY_OP( SIN, sin );
DEF_ELEMENT_WISE_UNARY_OP( COS, cos );
DEF_ELEMENT_WISE_UNARY_OP( EXP, exp );
DEF_ELEMENT_WISE_UNARY_OP( LOG, log );
DEF_ELEMENT_WISE_UNARY_OP( ELU, elu );
DEF_ELEMENT_WISE_UNARY_OP( NEG, neg );
DEF_ELEMENT_WISE_UNARY_OP( HARD_SIGMOID, hard_sigmoid );
DEF_ELEMENT_WISE_UNARY_OP( MISH, mish );
DEF_ELEMENT_WISE_UNARY_OP( ROUND, round );
DEF_ELEMENT_WISE_UNARY_OP( GELU, gelu );
DEF_ELEMENT_WISE_UNARY_OP( SELU, selu );
DEF_ELEMENT_WISE_UNARY_OP( CELU, celu );

#undef DEF_ELEMENT_UNARY_WISE_OP

#ifdef __cplusplus
}
#endif

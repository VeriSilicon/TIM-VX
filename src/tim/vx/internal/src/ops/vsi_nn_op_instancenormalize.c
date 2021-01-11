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
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

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
    vsi_nn_kernel_node_t    n = NULL;
    float eps = self->nn_param.instancenorm.eps;
    uint32_t *input_size = inputs[0]->attr.size;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    int32_t rs_flg = 0;

    param =vsi_nn_kernel_param_create();

    if((input_size[1] * input_size[2] < 65536)
        && dims_num > 2)
    {
        rs_flg = 1;
    }

    vsi_nn_kernel_param_add_float32( param, "eps", eps );
    vsi_nn_kernel_param_add_int32( param, "reshape_flg", rs_flg );
    n = vsi_nn_kernel_selector( self->graph, "instance_norm",
                    inputs, _INPUT_NUM, outputs, _OUTPUT_NUM, param );
    if( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if(param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
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
    BEGIN_IO_TYPE_DECL(INSTANCE_NORM, 3, 1)
        IO_TYPE(D_F16,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_F32,  D_F32,  D_F16,  D_F32)
        IO_TYPE(D_F32,  D_F32,  D_F32,  D_F32)
        IO_TYPE(D_I32,  D_F32,  D_F16,  D_I32)
        IO_TYPE(D_I32,  D_F32,  D_F16,  D_F32)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F16,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F16,  D_I16|Q_DFP)
    END_IO_TYPE_DECL(INSTANCE_NORM)
    if (!VALIDATE_OP_IO_TYPES(INSTANCE_NORM, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.instancenorm.lcl2_data =
    (vsi_nn_instancenorm_lcl_data2 *)malloc(sizeof(vsi_nn_instancenorm_lcl_data2));
    if (NULL == self->nn_param.instancenorm.lcl2_data)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset( self->nn_param.instancenorm.lcl2_data, 0, sizeof(vsi_nn_instancenorm_lcl_data2) );

    self->nn_param.instancenorm.lcl2_data->reshapeFlg = 0;
    self->nn_param.instancenorm.lcl2_data->execute_on_sw = 0;
    self->nn_param.instancenorm.lcl2_data->hash_idx = 0;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_INSTANCENORM_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.instancenorm.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.instancenorm.local.local_tensor[i]));
            self->nn_param.instancenorm.local.local_tensor[i] = NULL;
        }
    }
    if(self->nn_param.instancenorm.lcl2_data)
    {
        free(self->nn_param.instancenorm.lcl2_data);
        self->nn_param.instancenorm.lcl2_data = NULL;
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ INSTANCE_NORM,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif


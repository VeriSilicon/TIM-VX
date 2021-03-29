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

static vsi_status _try_set_high_presision_tensor
    (
    vsi_nn_tensor_t **inputs
    )
{
    vsi_status status;
    vsi_nn_vxtensor_attr_t attr;

    status = VSI_SUCCESS;
    attr = VSI_NN_TENSOR_ATTR_HIGH_PRECISION;

    if(VSI_NN_TYPE_FLOAT32 == inputs[1]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[1], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[2]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[2], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }

    return status;
}

static vsi_bool _is_3d_instance_norm
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs
    )
{
    if( 3 == inputs[0]->attr.dim_num )
    {
        return TRUE;
    }
    return FALSE;
} /* _is_3d_instance_norm() */

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
    vsi_nn_tensor_t * tmp_inputs[3]  = {NULL, NULL, NULL};
    vsi_nn_tensor_t * tmp_outputs[1] = {NULL};
    vsi_nn_instancenorm_lcl_data2 *local = self->nn_param.instancenorm.lcl2_data;

    status = _try_set_high_presision_tensor(inputs);
    if(status != VSI_SUCCESS)
    {
        VSILOGE("Set tensor attr of high presision fail");
        return status;
    }

    if(_is_3d_instance_norm(self, inputs))
    {
        tmp_inputs[0]  = local->reshaped_input;
        tmp_outputs[0] = local->reshaped_output;
        tmp_inputs[1] = inputs[1];
        tmp_inputs[2] = inputs[2];
    }
    else
    {
        tmp_inputs[0] = inputs[0];
        tmp_outputs[0] = outputs[0];
        tmp_inputs[1] = inputs[1];
        tmp_inputs[2] = inputs[2];
        if((input_size[1] * input_size[2] < 65536)
            && dims_num > 2)
        {
            rs_flg = 1;
        }
    }

    param =vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_float32( param, "eps", eps );
    vsi_nn_kernel_param_add_int32( param, "reshape_flg", rs_flg );
    n = vsi_nn_kernel_selector( self->graph, "instance_norm",
                    tmp_inputs, _INPUT_NUM, tmp_outputs, _OUTPUT_NUM, param );
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

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    uint32_t dim = 0;
    vsi_nn_instancenorm_lcl_data2 *local = NULL;
    uint32_t shape[VSI_NN_MAX_DIM_NUM];
    char tensor_name[128];

    dim = inputs[0]->attr.dim_num;
    if(_is_3d_instance_norm(self, inputs) == FALSE)
    {
        return VSI_SUCCESS;
    }

    VSILOGD("Optimize 3D %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
    /*
        insert a reshape node before and after 3D instance_norm
    */
    shape[0] = 1;
    shape[1] = inputs[0]->attr.size[0];
    shape[2] = inputs[0]->attr.size[1];
    shape[3] = inputs[0]->attr.size[2];
    dim = 4;
    local = self->nn_param.instancenorm.lcl2_data;
    if (VSI_NN_OPTIMIZE_FORWARD == direction)
    {
        /* reshape 3d input (xcn) --> 4d input (whcn) */
        local->reshaped_input = vsi_nn_reshape_tensor(self->graph, inputs[0], shape, dim);
    }
    else
    {
        /* reshape 3d output(xcn) --> 4d output(whcn) */
        local->reshaped_output = vsi_nn_reshape_tensor(self->graph, outputs[0], shape, dim);
        if(local->reshaped_output && local->reshaped_output->t)
        {
            memset(tensor_name, 0, sizeof(tensor_name));
            snprintf(tensor_name, sizeof(tensor_name), "uid_%u_reshape_out_0", self->uid);
            if(vxSetReferenceName((vx_reference)local->reshaped_output->t, tensor_name) == VSI_FAILURE)
            {
                VSILOGW("Set uid %u batchnorm reshaped output name fail", self->uid);
                return VSI_FAILURE;
            }
        }
    }

    return VSI_SUCCESS;
} /* op_optimize() */

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
    self->nn_param.instancenorm.lcl2_data->reshaped_input = NULL;
    self->nn_param.instancenorm.lcl2_data->reshaped_output = NULL;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    vsi_nn_instancenormalize_param *p = &(self->nn_param.instancenorm);
    for (i = 0; i < _VSI_NN_INSTANCENORM_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.instancenorm.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.instancenorm.local.local_tensor[i]));
            self->nn_param.instancenorm.local.local_tensor[i] = NULL;
        }
    }
    if(p->lcl2_data->reshaped_input)
    {
        vsi_nn_ReleaseTensor(&(p->lcl2_data->reshaped_input));
        p->lcl2_data->reshaped_input = NULL;
    }
    if(p->lcl2_data->reshaped_output)
    {
        vsi_nn_ReleaseTensor(&(p->lcl2_data->reshaped_output));
        p->lcl2_data->reshaped_output = NULL;
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
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif


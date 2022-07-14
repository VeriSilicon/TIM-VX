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
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_bool _is_pool1d
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs
    )
{
    /*
        support pool1d from version 1.1.31
    */
    if (vsi_nn_compareVersion(self->graph, 1, 1, 31) == -1)
    {
        return FALSE;
    }
    else
    {
        if ( 3 == inputs[0]->attr.dim_num )
        {
            return TRUE;
        }
        else
        {
            return FALSE;
        }
    }
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_pooling_params_ext_t params;
    vsi_nn_tensor_t * tmp_inputs[1]  = {NULL};
    vsi_nn_tensor_t * tmp_outputs[1] = {NULL};
    vsi_nn_pool_lcl_data *local = self->nn_param.pool.local;

    status = VSI_FAILURE;

    memset( &params, 0, sizeof( params ) );
    if (_is_pool1d(self, inputs))
    {
        // pool1d
        tmp_inputs[0]  = local->reshaped_input;
        tmp_outputs[0] = local->reshaped_output;

        params.base.pool_type = self->nn_param.pool.type;
        params.base.pool_size_x = self->nn_param.pool.ksize[0];
        params.base.pool_size_y = 1;
        params.base.pool_pad_x_left = self->nn_param.pool.pad[0];
        params.base.pool_pad_x_right = self->nn_param.pool.pad[1];
        params.base.pool_pad_y_top = 0;
        params.base.pool_pad_y_bottom = 0;
        params.base.rounding = self->vx_param.down_scale_size_rounding;
        params.stride_x = self->nn_param.pool.stride[0];
        params.stride_y = 1;
    }
    else
    {
        tmp_inputs[0] = inputs[0];
        tmp_outputs[0] = outputs[0];

        params.base.pool_type = self->nn_param.pool.type;
        params.base.pool_size_x = self->nn_param.pool.ksize[0];
        params.base.pool_size_y = self->nn_param.pool.ksize[1];
        params.base.pool_pad_x_left = self->nn_param.pool.pad[0];
        params.base.pool_pad_x_right = self->nn_param.pool.pad[1];
        params.base.pool_pad_y_top = self->nn_param.pool.pad[2];
        params.base.pool_pad_y_bottom = self->nn_param.pool.pad[3];
        params.base.rounding = self->vx_param.down_scale_size_rounding;
        params.stride_x = self->nn_param.pool.stride[0];
        params.stride_y = self->nn_param.pool.stride[1];
    }

    self->n = vxPoolingLayer2(
        self->graph->g,
        tmp_inputs[0]->t,
        (vx_nn_pooling_params_t *)&params,
        sizeof( params ),
        tmp_outputs[0]->t
        );

    if ( NULL != self->n )
    {
        status = VSI_SUCCESS;
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
    vsi_nn_pool_lcl_data *local = NULL;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM];
    char tensor_name[128];

    dim = inputs[0]->attr.dim_num;
    if(FALSE == _is_pool1d(self, inputs))
    {
        return VSI_SUCCESS;
    }

    VSILOGD("Optimize pool1d %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
    /*
        insert a reshape node before and after pool1d
    */
    local = self->nn_param.pool.local;
    if (VSI_NN_OPTIMIZE_FORWARD == direction)
    {
        /* reshape 3d input (xcn) --> 4d input (whcn) */
        shape[0] = inputs[0]->attr.size[0];//width
        shape[1] = 1;//height
        shape[2] = inputs[0]->attr.size[1];
        shape[3] = inputs[0]->attr.size[2];
        dim = 4;
        local->reshaped_input = vsi_nn_reshape_tensor(self->graph, inputs[0], shape, dim);
    }
    else
    {
        /* reshape 3d output(xcn) --> 4d output(whcn) */
        shape[0] = outputs[0]->attr.size[0];//width
        shape[1] = 1;//height
        shape[2] = outputs[0]->attr.size[1];
        shape[3] = outputs[0]->attr.size[2];
        dim = 4;
        local->reshaped_output = vsi_nn_reshape_tensor(self->graph, outputs[0], shape, dim);
        if (local->reshaped_output && local->reshaped_output->t)
        {
            memset(tensor_name, 0, sizeof(tensor_name));
            snprintf(tensor_name, sizeof(tensor_name), "uid_%u_reshape_out_0", self->uid);
            if (vxSetReferenceName((vx_reference)local->reshaped_output->t, tensor_name) == VSI_FAILURE)
            {
                VSILOGW("Set uid %u pool1d reshaped output name fail", self->uid);
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
    /* check inputs outputs data type */
    BEGIN_IO_TYPE_DECL(POOL, 1, 1)
        /* IO_TYPE(INPUT, OUTPUT) */
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_F32,          D_F16)
        IO_TYPE(D_F16,          D_F32)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_F16,          D_U8|Q_ASYM)
        IO_TYPE(D_F16,          D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_I8|Q_ASYM)
        IO_TYPE(D_F16,          D_I8|Q_SYM)
        IO_TYPE(D_F16,          D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_I16|Q_ASYM)
        IO_TYPE(D_F16,          D_I16|Q_SYM)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_SYM,     D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,    D_F16)
        IO_TYPE(D_I16|Q_ASYM,   D_F16)
        IO_TYPE(D_I16|Q_SYM,    D_F16)

        /* HW 9.0 */
        IO_TYPE(D_U8|Q_ASYM,    D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,    D_BF16)
        IO_TYPE(D_U8|Q_ASYM,    D_F32)
        IO_TYPE(D_U8|Q_ASYM,    D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,     D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_BF16)
        IO_TYPE(D_I8|Q_DFP,     D_F32)
        IO_TYPE(D_I16|Q_DFP,    D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,    D_BF16)
        IO_TYPE(D_I16|Q_DFP,    D_F32)
        IO_TYPE(D_F32,          D_BF16)
        IO_TYPE(D_BF16,         D_F32)
        IO_TYPE(D_F16,          D_BF16)
    END_IO_TYPE_DECL(POOL)
    if (!VALIDATE_OP_IO_TYPES(POOL, self, inputs, self->input.num, outputs, self->output.num)) {
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

    self->nn_param.pool.local =
    (vsi_nn_pool_lcl_data *)malloc(sizeof(vsi_nn_pool_lcl_data));
    if (NULL == self->nn_param.pool.local)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset( self->nn_param.pool.local, 0, sizeof(vsi_nn_pool_lcl_data) );

    self->nn_param.pool.local->reshaped_input = NULL;
    self->nn_param.pool.local->reshaped_output = NULL;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_pool_param *p = &(self->nn_param.pool);

    vsi_safe_release_tensor(p->local->reshaped_input);
    vsi_safe_release_tensor(p->local->reshaped_output);
    vsi_nn_safe_free(self->nn_param.pool.local);

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    vsi_size_t ksize[_cnt_of_array(self->nn_param.pool.ksize)] = {0}, i = 0;
    vsi_size_t pad[_cnt_of_array(self->nn_param.pool.pad)] = {0};

    ret = TRUE;

    for (i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
    {
        ksize[i] = self->nn_param.pool.ksize[i];
    }
    for (i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
    {
        pad[i] = self->nn_param.pool.pad[i];
    }
    if (_is_pool1d(self, inputs))
    {
        vsi_nn_compute_padding_conv1d(
            inputs[0]->attr.size,
            ksize,
            self->nn_param.pool.stride,
            NULL,
            self->nn_param.pool.pad_type,
            pad
        );
        for (i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
        {
            self->nn_param.pool.ksize[i] = (uint32_t)ksize[i];
        }
        for (i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
        {
            self->nn_param.pool.pad[i] = (uint32_t)pad[i];
        }

        /* Pooling */
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            self->nn_param.pool.ksize[0],
            &self->nn_param.pool.pad[0],
            self->nn_param.pool.stride[0],
            0,
            self->nn_param.pool.round_type
            );

        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
    }
    else
    {
        vsi_nn_compute_padding(
            inputs[0]->attr.size,
            ksize,
            self->nn_param.pool.stride,
            NULL,
            self->nn_param.pool.pad_type,
            pad
        );
        for (i = 0; i < _cnt_of_array(self->nn_param.pool.ksize); i++)
        {
            self->nn_param.pool.ksize[i] = (uint32_t)ksize[i];
        }
        for (i = 0; i < _cnt_of_array(self->nn_param.pool.pad); i++)
        {
            self->nn_param.pool.pad[i] = (uint32_t)pad[i];
        }

        /* Pooling */
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            self->nn_param.pool.ksize[0],
            &self->nn_param.pool.pad[0],
            self->nn_param.pool.stride[0],
            0,
            self->nn_param.pool.round_type
            );

        outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[1],
            self->nn_param.pool.ksize[1],
            &self->nn_param.pool.pad[2],
            self->nn_param.pool.stride[1],
            0,
            self->nn_param.pool.round_type
            );

        for (i = 2; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
    }

    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ POOL,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

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
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_util.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_FCL2_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_fcl_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.fcl);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, axis );
    //_SET_PARAM( 1, VX_TYPE_FLOAT32, bias );
    //_SET_PARAM( 2, VX_TYPE_TENSOR, data_bias );
    //_SET_PARAM( 3, VX_TYPE_TENSOR, data_weight );
    //_SET_PARAM( 4, VX_TYPE_FLOAT32, regularize );
    _SET_PARAM( 1, VX_TYPE_INT32, weights );
#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    uint32_t num
    )
{
    uint32_t i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t axis;
    vsi_nn_fcl_param * p;
    uint32_t i = 0;
    uint32_t num_fc = 1, num_no_fc = 1;
    uint32_t       num_of_dims[3]  = {0};
    uint32_t       input_size[VSI_NN_MAX_DIM_NUM]   = {0};
    uint32_t       output_size[VSI_NN_MAX_DIM_NUM]  = {0};
    uint32_t       weights_size[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t        size[VSI_NN_MAX_DIM_NUM]         = {0};
    uint32_t       ofm             = 0;
    uint32_t       dims            = 0;
    vx_tensor       input           = NULL;
    vx_tensor       output          = NULL;
    vx_tensor       weight          = NULL;
    vx_tensor       bias            = NULL;
    int32_t index = 0;
    vx_border_t border;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    p = (vsi_nn_fcl_param *)&(self->nn_param.fcl);
    axis = p->axis;

    memcpy(input_size, inputs[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_dims[0] = inputs[0]->attr.dim_num;
    memcpy(output_size, outputs[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_dims[1] = outputs[0]->attr.dim_num;
    memcpy(weights_size, inputs[1]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_dims[2] = inputs[1]->attr.dim_num;

    ofm = weights_size[num_of_dims[2] - 1];

    for(i = 0; i <= (uint32_t)axis; ++i)
    {
        num_fc *= input_size[i];
    }
    for(i = axis + 1; i < num_of_dims[0]; ++i)
    {
        num_no_fc *= input_size[i];
    }

    size[0] = num_fc;
    size[1] = num_no_fc;
    dims= 2;
    input = vxReshapeTensor(inputs[0]->t, size, dims);

    size[0] = num_fc;
    size[1] = ofm;
    dims= 2;
    weight = vxReshapeTensor(inputs[1]->t, size, dims);

    size[0] = ofm;
    size[1] = 1;
    dims= 2;
    bias = vxReshapeTensor(inputs[2]->t, size, dims);

    size[0] = ofm;
    size[1] = num_no_fc;
    dims= 2;
    output = vxReshapeTensor(outputs[0]->t, size, dims);

    status |= vxSetParameterByIndex(self->n, index++, (vx_reference)input);
    status |= vxSetParameterByIndex(self->n, index++, (vx_reference)weight);
    status |= vxSetParameterByIndex(self->n, index++, (vx_reference)bias);
    status |= vxSetParameterByIndex(self->n, index++, (vx_reference)output);

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.S16 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    if (input)  vxReleaseTensor(&input);
    if (weight) vxReleaseTensor(&weight);
    if (bias)   vxReleaseTensor(&bias);
    if (output) vxReleaseTensor(&output);

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    NULL
};
#endif

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
#if (USE_OVX_API == TRUE)
    uint32_t axis;
    vsi_nn_fcl_param * p;
    uint32_t i = 0;
    uint32_t num_fc = 1, num_no_fc = 1;
    uint32_t num_of_dims[4] = {0};
    int32_t input_size[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t output_size[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t weights_size[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t bias_size[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t ofm = 0;
    uint32_t dims = 0;
    vx_tensor input = NULL;
    vx_tensor output = NULL;
    vx_tensor weight = NULL;
    vx_tensor bias = NULL;

    p = (vsi_nn_fcl_param *)&(self->nn_param.fcl);
    axis = p->axis;

    memcpy(input_size, inputs[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_dims[0] = inputs[0]->attr.dim_num;
    memcpy(output_size, outputs[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_dims[1] = outputs[0]->attr.dim_num;
    memcpy(weights_size, inputs[1]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_dims[2] = inputs[1]->attr.dim_num;
    if( inputs[2] != NULL )
    {
        memcpy(bias_size, inputs[2]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
        num_of_dims[3] = inputs[2]->attr.dim_num;
    }

    ofm = weights_size[num_of_dims[2] - 1];

    for(i = 0; i <= (uint32_t)axis; ++i)
    {
        num_fc *= input_size[i];
    }
    for(i = axis + 1; i < num_of_dims[0]; ++i)
    {
        num_no_fc *= input_size[i];
    }

    input_size[0] = num_fc;
    input_size[1] = num_no_fc;
    dims= 2;
    input = vxReshapeTensor(inputs[0]->t, input_size, dims);

    weights_size[0] = num_fc;
    weights_size[1] = ofm;
    dims= 2;
    weight = vxReshapeTensor(inputs[1]->t, weights_size, dims);

    if( inputs[2] != NULL )
    {
        bias_size[0] = ofm;
        bias_size[1] = 1;
        dims= 2;
        bias = vxReshapeTensor(inputs[2]->t, bias_size, dims);
    }

    output_size[0] = ofm;
    output_size[1] = num_no_fc;
    dims= 2;
    output = vxReshapeTensor(outputs[0]->t, output_size, dims);

    self->n = vxFullyConnectedLayer(
        self->graph->g,
        input,
        weight,
        bias,
        self->vx_param.overflow_policy,
        self->vx_param.rounding_policy,
        output
        );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    if (input)  vxReleaseTensor(&input);
    if (weight) vxReleaseTensor(&weight);
    if (bias)   vxReleaseTensor(&bias);
    if (output) vxReleaseTensor(&output);
#else
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_fullconnect2";
    kernel_info.type = VX_KERNEL_TYPE_VX;
    kernel_info.kernel = vx_kernel_FCL2_list;
    kernel_info.kernel_index = 1;
    kernel_info.init_index = 1;
    self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
    if (kernel_info.resource_name) free(kernel_info.resource_name);
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
    }
#endif
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

    ret = vsi_nn_OpCheck(VSI_NN_OP_FCL, self, inputs, outputs);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_fcl_param * p;
    uint32_t i, j;
    uint32_t num_in_fmp = 1;

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        p = (vsi_nn_fcl_param *)&(self->nn_param.fcl);
        if (inputs[1]->attr.is_const == TRUE)
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num - p->axis;
            for(i = p->axis + 1, j = 1; i < inputs[0]->attr.dim_num && j < outputs[0]->attr.dim_num; ++i, ++j)
            {
                outputs[0]->attr.size[j] = inputs[0]->attr.size[i];
            }
        }
        else
        {
            /* For fullconnect_op, weight not const tensor */
            outputs[0]->attr.dim_num = 2;
            for (i = p->axis + 1; i < inputs[0]->attr.dim_num; i++)
            {
                num_in_fmp *= inputs[0]->attr.size[i];
            }
            outputs[0]->attr.size[1] = num_in_fmp;
        }
        outputs[0]->attr.size[0] = p->weights;
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ FCL2,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif


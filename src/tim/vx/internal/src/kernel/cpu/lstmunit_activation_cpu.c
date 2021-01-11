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


#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (LSTMUNIT_ACT_INPUTS_COUNT)
#define _OUTPUT_NUM         (LSTMUNIT_ACT_OUTUTS_COUNT)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.lstmunit_activation")


/*
 * Kernel params
 */
static vx_param_description_t _lstmunit_activation_kernel_param_def[] =
{
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*0  input_fc_i */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*1  input_fc_f */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*2  input_fc_c */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*3  input_fc_o */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*4  cs_in */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*5  hstate_fc_i */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*6  hstate_fc_f */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*7  hstate_fc_c */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*8  hstate_fc_o */
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*9  biases_i*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*10 biases_f*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*11 biases_c*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*12 biases_o*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*13 ln_w_i*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*14 ln_w_f*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*15 ln_w_c*/
    { VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*16 ln_w_o*/
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*17 output*/
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*18 cs_out*/
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*19 hs_out*/
    { VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },  /*20 _is_ln*/
    { VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },  /*21 _is_cifg*/
    { VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },  /*22 _is_proj*/
    { VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },  /*23 _is_hybrid*/
    { VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },  /*24 recurrent_activation*/
    { VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },  /*25 forget_bias*/
};
#define _LSTMUNIT_ACTIVATION_PARAM_NUM  _cnt_of_array( _lstmunit_activation_kernel_param_def )

#define SCALAR_IS_LN               (20)
#define SCALAR_IS_CIFG             (21)
#define SCALAR_IS_PROG             (22)
#define SCALAR_IS_HYBRID           (23)
#define SCALAR_ACTIVATION          (24)
#define SCALAR_FORGET_BIAS         (25)

static float activationFunctor(float a, vsi_nn_activation_e act_)
{
    switch (act_)
    {
      case VSI_NN_ACT_NONE:
        return a;
      case VSI_NN_ACT_RELU:
        return a < 0.f ? 0.f : a;
      case VSI_NN_ACT_RELU6:
        return vsi_nn_max(0.f, vsi_nn_min(a, 6.f));
      case VSI_NN_ACT_TANH:
        return (float)tanh(a);
      case VSI_NN_ACT_SIGMOID:
        return (float)(1.0f / (1.0f + exp(-a)));
      case VSI_NN_ACT_HARD_SIGMOID:
          a = a * 0.2f + 0.5f;
        return vsi_nn_max(0.f, vsi_nn_min(a, 1.f));
      default:
        // TODO(aselle): More informative fatal error!
        exit(1);
    }
}

#define gcoMATH_Exp(X)        (float)(expf((X)))
#define gcoMATH_TangentH(X)   (float)(tanhf((X)))

/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_compute)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t input[_INPUT_NUM]   = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float *f32_in_buffer[_INPUT_NUM]   = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM] = {NULL};
    size_t   in_stride_size[_INPUT_NUM][VSI_NN_MAX_DIM_NUM]   = {{1}};
    size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    size_t   out_elements[_OUTPUT_NUM] = {0};
    size_t   out_bytes[_OUTPUT_NUM]    = {0};
    uint32_t i, b;
    int32_t  _is_ln                = 0;
    int32_t  _is_cifg              = 0;
    int32_t  _is_proj              = 0;
    int32_t  _is_hybrid            = 0;
    int32_t  recurrent_activation;
    vsi_nn_activation_e activation_mode;
    uint32_t n_batch               = 0;
    uint32_t n_cell                = 0;
    float    forget_bias;
    /* prepare data */
    for( i = 0; i < _INPUT_NUM; i++ )
    {
        input[i]   = (vsi_nn_kernel_tensor_t)param[i];
        if (input[i])
        {
            in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
            vsi_nn_kernel_tensor_attr_get_stride( in_attr[i], in_stride_size[i] );
            f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
            CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );
        }

    }

    for( i = 0; i < _OUTPUT_NUM; i++ )
    {
        output[i]   = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        if (output[i])
        {
            out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );
            vsi_nn_kernel_tensor_attr_get_stride( out_attr[i], out_stride_size[i] );
            out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
            out_bytes[i] = out_elements[i] * sizeof(float);
            f32_out_buffer[i] = (float *)malloc( out_bytes[i] );
            CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
            memset( f32_out_buffer[i], 0, out_bytes[i] );
        }
    }

    status = vsi_nn_kernel_scalar_read_int32( (vsi_nn_kernel_scalar_t)param[SCALAR_IS_LN], &_is_ln );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( (vsi_nn_kernel_scalar_t)param[SCALAR_IS_CIFG], &_is_cifg );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( (vsi_nn_kernel_scalar_t)param[SCALAR_IS_PROG], &_is_proj );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32( (vsi_nn_kernel_scalar_t)param[SCALAR_IS_HYBRID], &_is_hybrid );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_ACTIVATION], &recurrent_activation );
    CHECK_STATUS_FAIL_GOTO(status, final );
    activation_mode = (vsi_nn_activation_e)recurrent_activation;
    status = vsi_nn_kernel_scalar_read_float32( (vsi_nn_kernel_scalar_t)param[SCALAR_FORGET_BIAS], &forget_bias );
    CHECK_STATUS_FAIL_GOTO(status, final );

    n_cell  = in_attr[LSTMUNIT_ACT_CSTATE_IN]->shape->data[0];
    n_batch = in_attr[LSTMUNIT_ACT_CSTATE_IN]->shape->data[1];

    for (b = 0; b < n_batch; b ++)
    {
        for (i = 0; i < n_cell; i++)
        {
            uint32_t index = i + n_cell * b;
            float    data_i_t = 0;
            float    data_f_t = 0;
            float    data_g_t = 0;
            float    data_o_t = 0;
            float    data_c_t = 0;
            float    data_h_t = 0;

            data_i_t = _is_cifg ? 0 : f32_in_buffer[LSTMUNIT_ACT_INPUT_FC_I][index];
            data_f_t = f32_in_buffer[LSTMUNIT_ACT_INPUT_FC_F][index];
            data_g_t = f32_in_buffer[LSTMUNIT_ACT_INPUT_FC_C][index];
            data_o_t = f32_in_buffer[LSTMUNIT_ACT_INPUT_FC_O][index];
            data_c_t = f32_in_buffer[LSTMUNIT_ACT_CSTATE_IN][index];

            if (!_is_ln)
            {
                data_i_t += _is_cifg ? 0 : f32_in_buffer[LSTMUNIT_ACT_HSTATE_FC_I][index];
                data_f_t += f32_in_buffer[LSTMUNIT_ACT_HSTATE_FC_F][index];
                data_g_t += f32_in_buffer[LSTMUNIT_ACT_HSTATE_FC_C][index];
                data_o_t += f32_in_buffer[LSTMUNIT_ACT_HSTATE_FC_O][index];
            }

            if (!_is_cifg)
            {
                if (_is_ln)
                {
                    data_i_t *= f32_in_buffer[LSTMUNIT_ACT_LN_WI][i];
                    data_i_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BI][i];
                }
                else if (_is_hybrid)
                {
                    data_i_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BI][i];
                }
            }

            if (_is_ln)
            {
                data_f_t *= f32_in_buffer[LSTMUNIT_ACT_LN_WF][i];
                data_f_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BF][i];
                data_g_t *= f32_in_buffer[LSTMUNIT_ACT_LN_WC][i];
                data_g_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BC][i];
                data_o_t *= f32_in_buffer[LSTMUNIT_ACT_LN_WO][i];
                data_o_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BO][i];
            }
            else if (_is_hybrid)
            {
                data_f_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BF][i];
                data_g_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BC][i];
                data_o_t += f32_in_buffer[LSTMUNIT_ACT_DATA_BO][i];
            }

            data_f_t += forget_bias;
            data_f_t = activationFunctor(data_f_t, activation_mode);

            if (_is_cifg)
                data_i_t = 1 - data_f_t;
            else
                data_i_t = activationFunctor(data_i_t, activation_mode);
            data_g_t = gcoMATH_TangentH(data_g_t);
            data_o_t = activationFunctor(data_o_t, activation_mode);
            data_c_t = data_f_t * data_c_t + data_i_t * data_g_t;
            data_h_t = data_o_t * gcoMATH_TangentH(data_c_t);

            f32_out_buffer[LSTMUNIT_ACT_CSTATE_OUT][index] = data_c_t;
            f32_out_buffer[LSTMUNIT_ACT_OUTPUT][index]     = data_h_t;

            if (!_is_proj)
            {
                f32_out_buffer[LSTMUNIT_ACT_HSTATE_OUT][index] = data_h_t;
            }
        }
    }

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        if (output[i])
        {
            status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                    f32_out_buffer[i], out_elements[i] );
            CHECK_STATUS_FAIL_GOTO( status, final );
        }
    }

final:
    for (i = 0; i < _INPUT_NUM; i++)
    {
        if (f32_in_buffer[i])
        {
            free(f32_in_buffer[i]);
            f32_in_buffer[i] = NULL;
        }

        if (in_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &in_attr[i] );
        }
    }
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        if (f32_out_buffer[i])
        {
            free(f32_out_buffer[i]);
            f32_out_buffer[i] = NULL;
        }

        if (out_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &out_attr[i] );
        }
    }

    return status;

} /* _compute() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    )
{
    vsi_status status = VSI_FAILURE;

    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _lstmunit_activation_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _lstmunit_activation_kernel_param_def );
    status                   = VSI_SUCCESS;

    return status;

} /* _query_kernel() */


static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_LSTMUNIT_ACTIVATION_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t  _is_ln= 0;
    int32_t  _is_cifg= 0;
    int32_t  _is_proj= 0;
    int32_t  _is_hybrid= 0;
    int32_t  recurrent_activation;
    float    forget_bias;

    _is_ln               = vsi_nn_kernel_param_get_int32( params, "_is_ln" );
    _is_cifg             = vsi_nn_kernel_param_get_int32( params, "_is_cifg" );
    _is_proj             = vsi_nn_kernel_param_get_int32( params, "_is_proj" );
    _is_hybrid           = vsi_nn_kernel_param_get_int32( params, "_is_hybrid" );
    recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );
    forget_bias          = vsi_nn_kernel_param_get_float32(params, "forget_bias");

    status = _query_kernel( kernel, inputs, outputs );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _LSTMUNIT_ACTIVATION_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_IS_LN] = vsi_nn_kernel_scalar_create(
                    graph, I32, &_is_ln );
            node_params[SCALAR_IS_CIFG] = vsi_nn_kernel_scalar_create(
                    graph, I32, &_is_cifg );
            node_params[SCALAR_IS_PROG] = vsi_nn_kernel_scalar_create(
                    graph, I32, &_is_proj );
            node_params[SCALAR_IS_HYBRID] = vsi_nn_kernel_scalar_create(
                    graph, I32, &_is_hybrid );
            node_params[SCALAR_ACTIVATION] = vsi_nn_kernel_scalar_create(
                    graph, I32, &recurrent_activation );
            node_params[SCALAR_FORGET_BIAS] = vsi_nn_kernel_scalar_create(
                    graph, F32, &forget_bias );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _LSTMUNIT_ACTIVATION_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_IS_LN] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_IS_CIFG] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_IS_PROG] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_IS_HYBRID] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_ACTIVATION] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_FORGET_BIAS] );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( lstmunit_activation, _setup )


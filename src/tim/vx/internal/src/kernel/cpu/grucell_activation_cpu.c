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
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.grucell_activation")

/*
 * Kernel params
 */
static vx_param_description_t _grucell_activation_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GRUCELL_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_kernel_param_def )

#define _IO_COUNT_DEFAULT       (5)

static vx_param_description_t _grucell_activation_separated_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GRUCELL_ACTIVATION_SEPARATED_PARAM_NUM  _cnt_of_array( _grucell_activation_separated_kernel_param_def )
#define _IO_COUNT_SEPARATED (15)
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
    int32_t i = 0;
    int32_t batch = 0;
    int32_t hidden_units = 0;
    float * buffer[_IO_COUNT_DEFAULT] = { NULL };
    vsi_status status = VSI_FAILURE;
    vsi_nn_activation_e gate_activation;
    vsi_nn_activation_e candidate_activation;
    vsi_nn_kernel_tensor_t tensors[_IO_COUNT_DEFAULT] = { NULL };
    vsi_nn_kernel_tensor_attr_t* attr[_IO_COUNT_DEFAULT] = { NULL };

    tensors[0] = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1] = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2] = (vsi_nn_kernel_tensor_t)param[2];
    tensors[3] = (vsi_nn_kernel_tensor_t)param[3];
    tensors[4] = (vsi_nn_kernel_tensor_t)param[4];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    attr[3] = vsi_nn_kernel_tensor_attr_create( tensors[3] );
    attr[4] = vsi_nn_kernel_tensor_attr_create( tensors[4] );

    /* z{t_} */
    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );
    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input buffer fail.", final );
    buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create input buffer fail.", final );
    buffer[3] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[3], attr[3], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create input buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &gate_activation);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &candidate_activation);
    CHECK_STATUS_FAIL_GOTO(status, final);

    batch = attr[0]->shape->data[1];
    hidden_units = attr[0]->shape->data[0];

    for( i = 0; i < batch * hidden_units; i++ )
    {
        float zt = vsi_nn_activation(buffer[0][i], gate_activation);
        float ht_ = vsi_nn_activation(buffer[1][i], candidate_activation);
        float ht_1 = buffer[2][i];
        float ht = zt * (ht_1 - ht_) + ht_;

        buffer[3][i] = ht;
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
            buffer[3], batch * hidden_units );
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vsi_nn_kernel_tensor_write_from_float( tensors[4], attr[4],
            buffer[3], batch * hidden_units );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < 5; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
    }
    return status;
} /* _compute() */

DEF_KERNEL_EXECUTOR(_compute_separated)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    int32_t i = 0, j = 0;
    int32_t batch = 0;
    int32_t hidden_units = 0;
    float * buffer[_IO_COUNT_SEPARATED] = { NULL };
    vsi_status status = VSI_FAILURE;
    vsi_nn_activation_e gate_activation;
    vsi_nn_activation_e candidate_activation;
    vsi_bool use_cudnn_implementation;
    grucell_activation_input_layout_e input_layout = GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC;
    vsi_nn_kernel_tensor_t tensors[_IO_COUNT_SEPARATED] = { NULL };
    vsi_nn_kernel_tensor_attr_t* attr[_IO_COUNT_SEPARATED] = { NULL };
    float *i_r_base = NULL, *i_c_base = NULL, *i_u_base = NULL;
    float *r_r_base = NULL, *r_u_base = NULL, *r_c_base = NULL;
    float cond_reset = 0.f, cond_update = 0.f, cond_candidate = 0.f;
    float i_r = 0.f, i_u = 0.f, i_c = 0.f, r_r = 0.f, r_u = 0.f, r_c = 0.f;
    float bias_r = 0.f, bias_u = 0.f, bias_c = 0.f;
    float r = 0.f, u = 0.f, c = 0.f, state = 0.f;

    for(i = 0; i < _IO_COUNT_SEPARATED; i++)
    {
        tensors[i] = (vsi_nn_kernel_tensor_t)param[i];
        attr[i] = vsi_nn_kernel_tensor_attr_create( tensors[i] );
    }

    /* z{t_} */
    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input buffer fail.", final );
    buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    buffer[3] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[3], attr[3], TRUE );

    buffer[4] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[4], attr[4], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[4], "Create input buffer fail.", final );
    buffer[5] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[5], attr[5], TRUE );
    buffer[6] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[6], attr[6], TRUE );

    buffer[7] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[7], attr[7], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[7], "Create input buffer fail.", final );
    buffer[8] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[8], attr[8], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[8], "Create input buffer fail.", final );
    buffer[9] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[9], attr[9], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[9], "Create input buffer fail.", final );

    buffer[10] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[10], attr[10], TRUE );
    buffer[11] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[11], attr[11], TRUE );
    buffer[12] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[12], attr[12], TRUE );

    buffer[13] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[13], attr[13], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[13], "Create input buffer fail.", final );
    buffer[14] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[14], attr[14], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[14], "Create input buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[15], &gate_activation);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[16], &candidate_activation);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[17], &use_cudnn_implementation);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[18], &input_layout);
    CHECK_STATUS_FAIL_GOTO(status, final);

    if(GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC == input_layout)
    {
        batch = attr[1]->shape->data[1];
        hidden_units = attr[1]->shape->data[0];

        if(buffer[2] == NULL)
        {
            hidden_units = hidden_units / 3;
        }

        for( i = 0; i < batch; i++ )
        {
            float* input_base = buffer[0] + i * hidden_units;
            float* output_base = buffer[13] + i * hidden_units;

            if(buffer[2] == NULL)
            {
                float* input_fc_base = buffer[1] + i * hidden_units * 3;
                float* recurrent_fc_base = buffer[4] + i * hidden_units * 3;

                i_r_base = input_fc_base + 0 * hidden_units;
                i_u_base = input_fc_base + 1 * hidden_units;
                i_c_base = input_fc_base + 2 * hidden_units;

                r_r_base = recurrent_fc_base + 0 * hidden_units;
                r_u_base = recurrent_fc_base + 1 * hidden_units;
                r_c_base = recurrent_fc_base + 2 * hidden_units;
            }
            else
            {
                i_r_base = buffer[1] + i * hidden_units;
                i_u_base = buffer[2] + i * hidden_units;
                i_c_base = buffer[3] + i * hidden_units;
                r_r_base = buffer[4] + i * hidden_units;
                r_u_base = buffer[5] + i * hidden_units;
                r_c_base = buffer[6] + i * hidden_units;
            }

            for( j = 0; j < hidden_units; j++ )
            {
                cond_reset = buffer[10] ? buffer[10][j] : cond_reset;
                cond_update = buffer[11] ? buffer[11][j] : cond_update;
                cond_candidate = buffer[12] ? buffer[12][j] : cond_candidate;

                bias_r = buffer[7][j];
                bias_u = buffer[8][j];
                bias_c = buffer[9][j];

                i_r = i_r_base[j];
                i_u = i_u_base[j];
                i_c = i_c_base[j];

                r_r = r_r_base[j];
                r_u = r_u_base[j];
                r_c = r_c_base[j];

                r = vsi_nn_activation(i_r + cond_reset + r_r + bias_r, gate_activation);
                u = vsi_nn_activation(i_u + cond_update + r_u + bias_u, gate_activation);
                c = vsi_nn_activation(i_c + cond_candidate + r * (r_c + bias_c), candidate_activation);
                state = u * (input_base[j] - c) + c;

                output_base[j] = state;
            }
        }
    }
    else
    {
        vsi_bool input_transposed = FALSE;
        float* input_base = buffer[0];
        float* output_base = buffer[13];
        float* curr_input = NULL;
        float* curr_output = NULL;

        batch = attr[1]->shape->data[0];
        hidden_units = attr[1]->shape->data[1];

        if(buffer[2] == NULL)
        {
            hidden_units = hidden_units / 3;
            i_r_base = buffer[1] + 0 * hidden_units * batch;
            i_u_base = buffer[1] + 1 * hidden_units * batch;
            i_c_base = buffer[1] + 2 * hidden_units * batch;
            r_r_base = buffer[4] + 0 * hidden_units * batch;
            r_u_base = buffer[4] + 1 * hidden_units * batch;
            r_c_base = buffer[4] + 2 * hidden_units * batch;
        }
        else
        {
            i_r_base = buffer[1];
            i_u_base = buffer[2];
            i_c_base = buffer[3];
            r_r_base = buffer[4];
            r_u_base = buffer[5];
            r_c_base = buffer[6];
        }

        if(GRUCELL_ACTIVATION_INPUT_LAYOUT_INPUT_NC_FC_CN == input_layout)
        {
            input_transposed = FALSE;
        }
        else
        {
            input_transposed = TRUE;
        }

        for( i = 0; i < hidden_units; i++ )
        {
            cond_reset = buffer[10] ? buffer[10][i] : cond_reset;
            cond_update = buffer[11] ? buffer[11][i] : cond_update;
            cond_candidate = buffer[12] ? buffer[12][i] : cond_candidate;
            bias_r = buffer[7][i];
            bias_u = buffer[8][i];
            bias_c = buffer[9][i];

            for( j = 0; j < batch; j++ )
            {
                if(input_transposed)
                {
                    curr_input = &input_base[i * batch + j];
                    curr_output = &output_base[i * batch + j];
                }
                else
                {
                    curr_input = &input_base[j * hidden_units + i];
                    curr_output = &output_base[j * hidden_units + i];
                }

                i_r = i_r_base[i * batch + j];
                i_u = i_u_base[i * batch + j];
                i_c = i_c_base[i * batch + j];
                r_r = r_r_base[i * batch + j];
                r_u = r_u_base[i * batch + j];
                r_c = r_c_base[i * batch + j];

                r = vsi_nn_activation(i_r + cond_reset + r_r + bias_r, gate_activation);
                u = vsi_nn_activation(i_u + cond_update + r_u + bias_u, gate_activation);
                c = vsi_nn_activation(i_c + cond_candidate + r * (r_c + bias_c), candidate_activation);
                state = u * (*curr_input - c) + c;

                *curr_output = state;
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[13], attr[13],
            buffer[13], batch * hidden_units );
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vsi_nn_kernel_tensor_write_from_float( tensors[14], attr[14],
            buffer[13], batch * hidden_units );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < _IO_COUNT_SEPARATED; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
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
    vsi_nn_tensor_t * const * const outputs,
    int32_t gate_activation,
    int32_t candidate_activation,
    int32_t input_category,
    vsi_bool use_cudnn_implementation,
    int32_t* param_count,
    int32_t* input_count,
    int32_t* output_count
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    if(input_category == 0)
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
        kernel->info.function    = _compute;
        kernel->info.parameters  = _grucell_activation_kernel_param_def;
        kernel->info.numParams   = _cnt_of_array( _grucell_activation_kernel_param_def );
        *param_count = _GRUCELL_ACTIVATION_PARAM_NUM;
        *input_count = 3;
        *output_count = 2;
        status = VSI_SUCCESS;
    }
    else
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
        kernel->info.function    = _compute_separated;
        kernel->info.parameters  = _grucell_activation_separated_kernel_param_def;
        kernel->info.numParams   = _cnt_of_array( _grucell_activation_separated_kernel_param_def );
        *param_count = _GRUCELL_ACTIVATION_SEPARATED_PARAM_NUM;
        *input_count = 13;
        *output_count = 2;
        status = VSI_SUCCESS;
    }
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
    vsi_nn_kernel_node_param_t* node_params = NULL;
    vsi_nn_kernel_node_t node = NULL;
    int32_t i = 0;
    int32_t j = 0;
    int32_t param_count = 0;
    int32_t input_count = 0;
    int32_t output_count = 0;
    int32_t gate_activation = vsi_nn_kernel_param_get_int32( params, "gate_activation" );
    int32_t candidate_activation = vsi_nn_kernel_param_get_int32( params, "candidate_activation" );
    int32_t input_category = vsi_nn_kernel_param_get_int32( params, "input_category" );
    int32_t use_cudnn_implementation = vsi_nn_kernel_param_get_int32( params, "use_cudnn_implementation" );
    grucell_activation_input_layout_e input_layout = vsi_nn_kernel_param_get_int32( params, "input_layout" );
    vsi_nn_tensor_t** _inputs = NULL;

    status = _query_kernel( kernel, inputs, outputs, gate_activation, candidate_activation,
        input_category, use_cudnn_implementation, &param_count, &input_count, &output_count );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            _inputs = (vsi_nn_tensor_t**)malloc(input_count * sizeof(vsi_nn_tensor_t**));
            node_params = (vsi_nn_kernel_node_param_t *)malloc(sizeof(vsi_nn_kernel_node_param_t) * param_count);
            for(i = 0; i < input_count; i++)
            {
                _inputs[i] = inputs[i];
            }

            j = input_count + output_count;

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, param_count,
                    _inputs, input_count, outputs, output_count );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &gate_activation );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &candidate_activation );
            if(input_category != 0)
            {
                node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &use_cudnn_implementation );
                node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &input_layout );
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, param_count );
            if(input_category != 0)
            {
                vsi_nn_kernel_scalar_release( &node_params[--j] );
                vsi_nn_kernel_scalar_release( &node_params[--j] );
            }
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
        }
    }

    vsi_nn_safe_free(_inputs);
    vsi_nn_safe_free(node_params);
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( grucell_activation, _setup )


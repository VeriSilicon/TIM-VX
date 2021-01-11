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
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"
#include "client/vsi_nn_vxkernel.h"

__BEGIN_DECLS

/** Unary Kernel internal type */
typedef enum
{
    UNARY_SIN,
    UNARY_EXP,
    UNARY_LOG,
    UNARY_ELU,
    UNARY_NEG,
    UNARY_HSIGMOID,
    UNARY_MISH,
} unary_type_e;


#define _CPU_ARG_NUM            (1)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("eltwise_unary_sw")

static float exp_eval(float data)
{
    return expf(data);
}

static float sin_eval(float data)
{
    return sinf(data);
}

static float log_eval(float data)
{
    return logf(data);
}

static float elu_eval(float data)
{
    return data >=0 ? data : expf(data) - 1;
}

static float neg_eval(float data)
{
    return data * -1.0f;
}

static float hsigmoid_eval(float data)
{
    data = (float)(0.2 * data + 0.5);
    data = vsi_nn_clamp(data, 0, 1);

    return data;
}

static float soft_plus_eval(float data)
{
    return log_eval(exp_eval(data) + 1);
}

static float mish_eval(float data)
{
    data = (float)(data * tanh(soft_plus_eval(data)));

    return data;
}

DEF_KERNEL_EXECUTOR(_eltwise_unary_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    int32_t i;
    int32_t unary_type = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &unary_type);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );
    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    for ( i = 0; i < (int32_t)out_elements; ++i)
    {
        float data = buffer[0][i];

        switch (unary_type)
        {
        case UNARY_SIN:
            data = sin_eval(data);
            break;
        case UNARY_EXP:
            data = exp_eval(data);
            break;
        case UNARY_LOG:
            data = log_eval(data);
            break;
        case UNARY_ELU:
            data = elu_eval(data);
            break;
        case UNARY_NEG:
            data = neg_eval(data);
            break;
        case UNARY_HSIGMOID:
            data = hsigmoid_eval(data);
            break;
        case UNARY_MISH:
            data = mish_eval(data);
            break;
        default:
            break;
        }
        buffer[1][i] = (float)data;
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
            buffer[i] = NULL;
        }
    }
    return status;
} /* _eltwise_unary_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define INPUT_FUNC_TYPE           (2)

static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _eltwise_unary_exec,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
    return VSI_SUCCESS;
} /* _query_kernel() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel,
    const unary_type_e            unary_type
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            backend_params[INPUT_FUNC_TYPE] = vsi_nn_kernel_scalar_create(
                    graph, I32, &unary_type );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &backend_params[INPUT_FUNC_TYPE] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }

    return node;
} /* _setup() */

#define REGISTER_ELTWISE_UNARY_BACKEND_CPU(KERNEL_NAME, UNARY_TYPE) \
    static vsi_nn_kernel_node_t _##KERNEL_NAME##_setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num, \
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ) \
    { \
        return _setup(graph, inputs, input_num, outputs, output_num, \
                params, kernel, UNARY_TYPE); \
    } \
    REGISTER_BACKEND_CPU( KERNEL_NAME, _##KERNEL_NAME##_setup )

__END_DECLS

REGISTER_ELTWISE_UNARY_BACKEND_CPU( sin,          UNARY_SIN )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( exp,          UNARY_EXP )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( log,          UNARY_LOG )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( elu,          UNARY_ELU )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( neg,          UNARY_NEG )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( hard_sigmoid, UNARY_HSIGMOID )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( mish,         UNARY_MISH )

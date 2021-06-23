/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "repeat"

// Add kernel hashtable here

#define HASH_REPEAT_KERNEL_NAME(SRC0_TYPE, AXIS) \
    CVIVANTE_NAMESPACE("cl.repeat_"#SRC0_TYPE"_axis"#AXIS)

#define HASH_REPEAT_KERNEL_1D_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("cl.repeat_"#SRC0_TYPE"_1D")

// Add kernel hashtable here
#define HASH_REPEAT_KEY(_input0_type, _output_type, _is1d, _axis) \
    ((_input0_type << 24) | (_output_type << 16) | (_is1d << 8) | _axis)

#define TENSOR_REPEAT_KERNELS(IN0_TYPE, OUT_TYPE, AXIS, SOURCE) \
    { HASH_REPEAT_KEY(IN0_TYPE, OUT_TYPE, 0, AXIS), \
        HASH_REPEAT_KERNEL_NAME(IN0_TYPE, AXIS), \
        SOURCE },

#define TENSOR_REPEAT_1D_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_REPEAT_KEY(IN0_TYPE, OUT_TYPE, 1, 0), \
        HASH_REPEAT_KERNEL_1D_NAME(IN0_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _repeat_kernel_map[] =
{
    // Register kernel here
    TENSOR_REPEAT_KERNELS( I32, I32, 0, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( I32, I32, 1, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( I32, I32, 2, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( F32, F32, 0, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( F32, F32, 1, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( F32, F32, 2, KERNEL_SOURCE_1 )

    TENSOR_REPEAT_1D_KERNELS( I32,  I32,  KERNEL_SOURCE_1 )
    TENSOR_REPEAT_1D_KERNELS( F32,  F32,  KERNEL_SOURCE_1 )
};

/*
 * Kernel params
 */
static vx_param_description_t _repeat_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _REPEAT_PARAM_NUM  _cnt_of_array( _repeat_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_repeat_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_int_array_t * input_shape = NULL;
    int32_t height = 0, width = 0, chn = 0;
    int32_t is1d = 0;
    int32_t axis = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input_shape  = attr[0]->shape;
    width = input_shape->data[0];
    height = input_shape->data[1];
    if (height == 1 && input_shape->size == 2)
    {
        is1d = 1;
    }
    chn = input_shape->size > 2 ? input_shape->data[2] : 1;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = width;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = chn;
    if (is1d || axis == 1)
    {
        gpu_param.global_size[0]   = 1;
    }
    else if (axis == 0)
    {
        gpu_param.global_size[1] = 1;
    }
    else if (axis == 2)
    {
        gpu_param.global_size[2] = 1;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _repeat_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    int32_t is1d = inputs[0]->attr.dim_num == 1 ? 1 : 0;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == F16)
    {
        input0_dtype = F32;
    }
    if (output_dtype == F16)
    {
        output_dtype = F32;
    }

    key = HASH_REPEAT_KEY( input0_dtype, output_dtype, is1d, axis );

    for( i = 0; i < _cnt_of_array(_repeat_kernel_map); i ++ )
    {
        if ( _repeat_kernel_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(_repeat_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _repeat_kernel_map[i].function_name );
        kernel->info.parameters = _repeat_kernel_param_def;
        kernel->info.numParams = _REPEAT_PARAM_NUM;
        kernel->info.initialize = _repeat_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                _repeat_kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _repeat_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static int32_t _optimize_repeat_shape
    (
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    int32_t* axis,
    int32_t* opt_shape_in,
    int32_t* opt_shape_out,
    int32_t* new_rank
    )
{
    vsi_status status = VSI_SUCCESS;

    if (inputs[0]->attr.dim_num == 1)
    {
        opt_shape_in[0] = inputs[0]->attr.size[0];
        opt_shape_in[1] = 1;
        opt_shape_out[0] = outputs[0]->attr.size[0];
        opt_shape_out[1] = 1;
        new_rank[0] = 2;
        new_rank[1] = 2;
    }
    else if (axis[0] == 3)
    {
        vsi_nn_kernel_optimize_element_shape( (int32_t*)inputs[0]->attr.size, 3, opt_shape_in, new_rank );
        if (opt_shape_in[1] == 1)
        {
            opt_shape_in[1] = inputs[0]->attr.size[3];
            opt_shape_out[0] = opt_shape_in[0];
            opt_shape_out[1] = outputs[0]->attr.size[3];
            axis[0] = 0;
            new_rank[0] = 2;
            new_rank[1] = 2;
        }
        else if (new_rank[0] == 2)
        {
            opt_shape_in[2] = inputs[0]->attr.size[3];
            opt_shape_out[0] = opt_shape_in[0];
            opt_shape_out[1] = opt_shape_in[1];
            opt_shape_out[2] = outputs[0]->attr.size[3];
            axis[0] = 2;
            new_rank[0] = 3;
            new_rank[1] = 3;
        }
        else
        {
            status = VSI_FAILURE;
        }
    }

    return status;
}


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
    vsi_nn_kernel_node_param_t node_params[_REPEAT_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_input1 = NULL, rs_output = NULL;
    int32_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }};
    int32_t new_rank[2] = {0, 0};
    int32_t axis  = vsi_nn_kernel_param_get_int32( params, "axis" );

    int32_t width = inputs[0]->attr.size[0];
    int32_t height = inputs[0]->attr.dim_num > 1 ? inputs[0]->attr.size[1] : 1;
    int32_t channel = inputs[0]->attr.dim_num > 2 ? inputs[0]->attr.size[2] : 1;

    if ( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if (axis > 2 || outputs[0]->attr.dim_num == 1)
    {
        status = _optimize_repeat_shape(inputs, outputs, &axis, new_shape[0], new_shape[1], new_rank);
        if ( VSI_SUCCESS != status )
        {
            goto final;
        }
        rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape[0], new_rank[0]);
        rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape[1], new_rank[1]);

        width = new_shape[0][0];
        height = new_shape[0][1];
        channel = new_rank[0] > 2 ? new_shape[0][2]: 1;
    }

    if (inputs[1]->attr.dim_num == 1)
    {
        new_shape[0][0] = inputs[1]->attr.size[0];
        new_shape[0][1] = 1;
        rs_input1 = vsi_nn_kernel_tensor_reshape(inputs[1]->t, new_shape[0], 2);
    }

    status = _query_kernel( kernel, inputs, outputs, axis );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    node = vsi_nn_kernel_create_node( graph, kernel );
    if (node)
    {
        uint32_t index = 0;
        if (rs_input)
        {
            node_params[index++] = rs_input;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[0]->t;
        }
        if (rs_input1)
        {
            node_params[index++] = rs_input1;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
        }
        if (rs_output)
        {
            node_params[index++] = rs_output;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)outputs[0]->t;
        }
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &height );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &channel );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis );

        status  = vsi_nn_kernel_node_pass_param( node, node_params,
            _REPEAT_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &node_params[3] );
        vsi_nn_kernel_scalar_release( &node_params[4] );
        vsi_nn_kernel_scalar_release( &node_params[5] );
        vsi_nn_kernel_scalar_release( &node_params[6] );
    }

    /* Pass parameters to node. */
final:
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_input1)
    {
        vsi_nn_kernel_tensor_release( &rs_input1 );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( repeat, _setup )


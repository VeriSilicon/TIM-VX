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

#define KERNEL_SOURCE_1    "cumsum"
#define KERNEL_SOURCE_2    "cumsum_2d"

// Add kernel hashtable here
#define HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, _image_2d) \
    ((AXIS << 20) | (IN_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

#define HASH_CUMSUM_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0), \
        CVIVANTE_NAMESPACE("cl.cumsum_"#IN_DTYPE"to"#OUT_DTYPE"_axis"#AXIS), \
        KERNEL_SOURCE_1 },

#define HASH_CUMSUM_KERNELS_2D( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_CUMSUM_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("cl.cumsum_"#IN_DTYPE"to"#OUT_DTYPE"_axis"#AXIS"_2D"), \
        KERNEL_SOURCE_2 },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } cumsum_map[] =
{
    HASH_CUMSUM_KERNELS(0, U8,  U8)
    HASH_CUMSUM_KERNELS(0, F32, F32)
    HASH_CUMSUM_KERNELS(0, F32, U8)
    HASH_CUMSUM_KERNELS(1, U8,  U8)
    HASH_CUMSUM_KERNELS(1, F32, F32)
    HASH_CUMSUM_KERNELS(1, F32, U8)
    HASH_CUMSUM_KERNELS(2, U8,  U8)
    HASH_CUMSUM_KERNELS(2, F32, F32)
    HASH_CUMSUM_KERNELS(2, F32, U8)
    HASH_CUMSUM_KERNELS_2D(0, U8,  U8)
    HASH_CUMSUM_KERNELS_2D(0, F32, F32)
    HASH_CUMSUM_KERNELS_2D(0, F32, U8)
    HASH_CUMSUM_KERNELS_2D(1, U8,  U8)
    HASH_CUMSUM_KERNELS_2D(1, F32, F32)
    HASH_CUMSUM_KERNELS_2D(1, F32, U8)
};

/*
 * Kernel params
 */
static vx_param_description_t _cumsum_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CUMSUM_PARAM_NUM  _cnt_of_array( _cumsum_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_cumsum_initializer)
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
    vsi_size_array_t * input_shape = NULL;
    int32_t       axis      = 0;
    int32_t       width     = 0;
    int32_t       height    = 0;
    int32_t       channel   = 0;
    int32_t       w         = 1;
    int32_t       h         = 1;
    int32_t       c         = 1;
    uint32_t      dim       = 1;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input_shape  = attr[0]->shape;
    dim     = (uint32_t)input_shape->size;
    width   = (int32_t)(input_shape->data[0]);
    height  = (int32_t)(input_shape->data[1]);
    channel = (int32_t)(dim > 2 ? input_shape->data[2] : 1);

    if (axis == 0)
    {
        w = 1;
        h = height;
        c = channel;
    }
    else if (axis == 1)
    {
        w = width;
        h = 1;
        c = channel;
    }
    else if (axis == 2)
    {
        w = width;
        h = height;
        c = 1;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = w;
    gpu_param.global_size[1]   = h;
    gpu_param.global_size[2]   = c;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _cumsum_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis,
    int32_t is_2d
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == U32)
    {
        input0_dtype = U8;
    }

    if (input0_dtype == F16)
    {
        input0_dtype = F32;
    }

    if (output_dtype == U32)
    {
        output_dtype = U8;
    }

    if (output_dtype == F16)
    {
        output_dtype = F32;
    }

    key = HASH_CUMSUM_HASH_KEY( axis, input0_dtype, output_dtype, is_2d);

    for ( i = 0; i < _cnt_of_array(cumsum_map); i ++ )
    {
        if ( cumsum_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(cumsum_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  cumsum_map[i].function_name );
        kernel->info.parameters = _cumsum_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _cumsum_kernel_param_def );
        kernel->info.initialize = _cumsum_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                cumsum_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                cumsum_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_CUMSUM_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[1][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    int32_t axis       = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t exclusive  = vsi_nn_kernel_param_get_int32( params, "exclusive" );
    int32_t reverse    = vsi_nn_kernel_param_get_int32( params, "reverse" );
    int32_t axis_new   = 0;
    int32_t is_2d      = 0;
    uint32_t rs_dim    = 2;
    int32_t input_zp   = vsi_nn_get_tensor_zero_point(inputs[0]);
    float input_scale  = vsi_nn_get_tensor_scale(inputs[0]);
    float output_zp    = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float in_out_scale = input_scale * output_scale;
    float in_out_zp_scale = in_out_scale * input_zp;
    int32_t width      = 0;
    int32_t height     = 0;
    int32_t channel    = 1;
    int32_t i = 0;

    vsi_nn_kernel_optimize_softmax_shape(
                inputs[0]->attr.size, inputs[0]->attr.dim_num, axis,
                shapes[0], &rs_dim, &axis_new);
    if (rs_dim > 3)
    {
        return NULL;
    }

    width = (int32_t)shapes[0][0];
    height = (int32_t)shapes[0][1];

    if (rs_dim == 2)
    {
        is_2d = 1;
    }
    else
    {
        channel = (int32_t)shapes[0][2];
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], (vsi_size_t)rs_dim );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[0], (vsi_size_t)rs_dim );

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, axis_new, is_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 2;

            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _CUMSUM_PARAM_NUM,
                reshape_tensors, 1, &reshape_tensors[1], 1 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_new );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &exclusive );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &height );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &channel );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &input_zp );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &in_out_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &in_out_zp_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &output_zp );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CUMSUM_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
        }
    }

    for (i = 0; i < 2; i++)
    {
        vsi_safe_release_tensor(reshape_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( cumsum, _setup )

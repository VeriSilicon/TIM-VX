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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

#define KERNEL_SOURCE_1    "layer_normalization"

#define HASH_LAYERNORM_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.layer_norm_"#SRC0_TYPE"to"#DST_TYPE)

// Add kernel hashtable here
#define HASH_LAYERNORM_KEY(_input0_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_output_type << 16) | (_reshape_flag << 8))

#define TENSOR_LAYERNORM_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_LAYERNORM_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _layernorm_kernel_map[] =
{
    // Register kernel here
    TENSOR_LAYERNORM_KERNELS( F32, F32, KERNEL_SOURCE_1 )
    TENSOR_LAYERNORM_KERNELS( U8,  U8, KERNEL_SOURCE_1 )
};

/*
 * Kernel params
 */
static vx_param_description_t _layernorm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _LAYERNORM_PARAM_NUM  _cnt_of_array( _layernorm_kernel_param_def )

/*
 * Kernel initializer
 */

DEF_KERNEL_INITIALIZER(_layernorm_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * input_shape = NULL;
    //int32_t width = 0;
    vsi_ssize_t height = 0;
    vsi_ssize_t chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    input_shape  = attr[0]->shape;
    //width = input_shape->data[0];
    height = input_shape->data[1];
    chn = (input_shape->size <= 2) ? 1 : input_shape->data[2];

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.local_size[0]    = 16;
    gpu_param.local_size[1]    = 1;
    gpu_param.local_size[2]    = 1;
    gpu_param.global_size[0]   = 16;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = chn;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    return status;
} /* _layernorm_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t* kernel,
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t reshape2D
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == F16 && output_dtype == F16)
    {
        input0_dtype = F32;
        output_dtype = F32;
    }

    key = HASH_LAYERNORM_KEY( input0_dtype, output_dtype, 0 );

    for( i = 0; i < _cnt_of_array(_layernorm_kernel_map); i ++ )
    {
        if ( _layernorm_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_layernorm_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _layernorm_kernel_map[i].function_name );
        kernel->info.parameters = _layernorm_kernel_param_def;
        kernel->info.numParams = _LAYERNORM_PARAM_NUM;
        kernel->info.initialize = _layernorm_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                _layernorm_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _layernorm_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_LAYERNORM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_gamma = NULL, rs_beta = NULL;

    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" );

    vsi_size_t width = inputs[0]->attr.size[0];
    vsi_size_t height = inputs[0]->attr.size[1];
    int32_t input_fl = 0;
    float input_zp = 0.0f;
    float input_scale = 1.0f;
    int32_t output_fl = 0;
    float output_zp = 0.0f;
    float output_scale = 1.0f;
    float e2InScale = 1.0f, scale_inOut = 1.0f;
    float dim_ratio = (float)1.0 / (float)(width);
    float sumZpScale = 0.0f;
    float zp2ScaleE2 = 0.0f;
    float sumZpScaleE2 = 0.0f;

    if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        input_zp = (float)inputs[0]->attr.dtype.zero_point;
        input_scale = inputs[0]->attr.dtype.scale;
    }
    else if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        input_fl = inputs[0]->attr.dtype.fl;
        if (input_fl > 0)
        {
            input_scale = (1.0f / ((float) ((int64_t)1 << input_fl)));
        }
        else
        {
            input_scale = ((float) ((int64_t)1 << -input_fl));
        }
        input_zp = 0.0f;
    }

    if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        output_zp = (float)outputs[0]->attr.dtype.zero_point;
        output_scale = 1.0f / outputs[0]->attr.dtype.scale;
    }
    else if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        output_fl = outputs[0]->attr.dtype.fl;
        if (output_fl > 0)
        {
            output_scale = (float)((int64_t)1 << output_fl);
        }
        else
        {
            output_scale = (1.0f / (float)((int64_t)1 << -output_fl));
        }
        output_zp = 0.0f;
    }
    scale_inOut = input_scale * output_scale;
    e2InScale = input_scale * input_scale;
    sumZpScale = width * input_zp * input_scale;
    zp2ScaleE2 = input_zp * 2 * e2InScale;
    sumZpScaleE2 = width * input_zp * input_zp * e2InScale;

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, 0 );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    if (inputs[1]->attr.dim_num < 2)
    {
        vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
        shape[0] = inputs[1]->attr.size[0];
        shape[1] = 1;
        shape[2] = 1;
        shape[3] = 1;
        rs_beta = vsi_nn_kernel_tensor_reshape( inputs[1]->t, shape, 4 );
    }
    if (inputs[2]->attr.dim_num < 2)
    {
        vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
        shape[0] = inputs[2]->attr.size[0];
        shape[1] = 1;
        shape[2] = 1;
        shape[3] = 1;
        rs_gamma = vsi_nn_kernel_tensor_reshape( inputs[2]->t, shape, 4 );
    }

    // Nomalization
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if (node)
        {
            uint32_t index = 0;
            node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[0]->t;
            if (inputs[1]->attr.dim_num < 2)
            {
                node_params[index++] = rs_beta;
            }
            else
            {
                node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
            }
            if (inputs[2]->attr.dim_num < 2)
            {
                node_params[index++] = rs_gamma;
            }
            else
            {
                node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
            }
            node_params[index++] = (vsi_nn_kernel_node_param_t)outputs[0]->t;
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_zp );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &output_zp );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &output_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &e2InScale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_inOut );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &sumZpScale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &zp2ScaleE2 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &sumZpScaleE2 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &height );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &dim_ratio );

            status  = vsi_nn_kernel_node_pass_param( node, node_params,
                        _LAYERNORM_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
            vsi_nn_kernel_scalar_release( &node_params[13] );
            vsi_nn_kernel_scalar_release( &node_params[14] );
            vsi_nn_kernel_scalar_release( &node_params[15] );
            vsi_nn_kernel_scalar_release( &node_params[16] );
        }
    }

    /* Pass parameters to node. */
final:
    if (rs_beta)
    {
        vsi_nn_kernel_tensor_release( &rs_beta );
    }
    if (rs_gamma)
    {
        vsi_nn_kernel_tensor_release( &rs_gamma );
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( layer_norm, _setup )


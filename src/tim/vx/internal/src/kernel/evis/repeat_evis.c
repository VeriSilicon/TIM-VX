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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

#define KERNEL_SOURCE_1    "repeat"
#define KERNEL_SOURCE_2    "repeat_axis1"

#define HASH_PREPROCESS_STARTID_SH_KERNEL_NAME \
    CVIVANTE_NAMESPACE("evis.preprocess_start_idx")

#define HASH_REPEAT_SH_KERNEL_1D_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("evis.repeat_"#SRC0_TYPE"_1D")

#define HASH_REPEAT_SH_KERNEL_NAME(SRC0_TYPE, AXIS) \
    CVIVANTE_NAMESPACE("evis.repeat_"#SRC0_TYPE"_axis"#AXIS)

// Add kernel hashtable here
#define HASH_PREPROCESS_KEY(_input0_type, _output_type) \
    ((_input0_type << 24) | (_output_type << 16))

#define HASH_REPEAT_KEY(_input0_type, _output_type, _is1d, _axis) \
    ((_input0_type << 24) | (_output_type << 16) | (_is1d << 8) | _axis)

#define TENSOR_PREPROCESS_STARTID_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_PREPROCESS_KEY(IN0_TYPE, OUT_TYPE), \
        HASH_PREPROCESS_STARTID_SH_KERNEL_NAME, \
        SOURCE },

#define TENSOR_REPEAT_KERNELS(IN0_TYPE, OUT_TYPE, AXIS, SOURCE) \
    { HASH_REPEAT_KEY(IN0_TYPE, OUT_TYPE, 0, AXIS), \
        HASH_REPEAT_SH_KERNEL_NAME(IN0_TYPE, AXIS), \
        SOURCE },

#define TENSOR_REPEAT_1D_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_REPEAT_KEY(IN0_TYPE, OUT_TYPE, 1, 0), \
        HASH_REPEAT_SH_KERNEL_1D_NAME(IN0_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _preprocess_kernel_map[] =
{
    // Register kernel here
    TENSOR_PREPROCESS_STARTID_KERNELS( I32, I32, KERNEL_SOURCE_1 )
};

static const _kernel_map_type _repeat_kernel_map[] =
{
    // Register kernel here
    TENSOR_REPEAT_KERNELS( U8,  U8,  0, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( U8,  U8,  1, KERNEL_SOURCE_2 )
    TENSOR_REPEAT_KERNELS( U8,  U8,  2, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( I16, I16, 0, KERNEL_SOURCE_1 )
    TENSOR_REPEAT_KERNELS( I16, I16, 1, KERNEL_SOURCE_2 )
    TENSOR_REPEAT_KERNELS( I16, I16, 2, KERNEL_SOURCE_1 )

    TENSOR_REPEAT_1D_KERNELS( U8,  U8,  KERNEL_SOURCE_1 )
    TENSOR_REPEAT_1D_KERNELS( I16, I16, KERNEL_SOURCE_1 )
};

/*
 * Kernel params
 */
static vx_param_description_t _preprocess_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _REPEAT_PREPROCESS_PARAM_NUM  _cnt_of_array( _preprocess_kernel_param_def )

static vx_param_description_t _repeat_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _REPEAT_PARAM_NUM  _cnt_of_array( _repeat_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_preprocess_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[1] = {NULL};
    int32_t width = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    width = attr[0]->shape->data[0];

    shaderParam.global_scale[0]  = 16;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.local_size[0]    = 32;
    shaderParam.local_size[1]    = 1;
    shaderParam.local_size[2]    = 1;
    shaderParam.global_size[0]   = 32;
    shaderParam.global_size[1]   = 1;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniIntegralHorAcc_4x4 = {{
                0xff3f0f03, // TCfg
                0x00000000, // ASelt
                0x00100000, 0x32100210, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniIntegralHorAcc_4x4", &uniIntegralHorAcc_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "width", &width);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_repeat_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[1] = {NULL};
    vsi_int_array_t * input_shape = NULL;
    int32_t height = 0, width = 0, chn = 0;
    int32_t is1d = 0;
    int32_t axis = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &axis);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_shape  = attr[0]->shape;
    width = input_shape->data[0];
    height = input_shape->data[1];
    if (height == 1 && input_shape->size == 2)
    {
        is1d = 1;
    }
    chn = input_shape->size > 2 ? input_shape->data[2] : 1;

    if ((axis == 0 && is1d == 0) || axis == 2)
    {
        shaderParam.global_scale[0]  = 16;
        if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
        {
            shaderParam.global_scale[0]  = 8;
        }

        shaderParam.global_scale[1]  = 1;
        shaderParam.global_scale[2]  = 1;
    }
    else if (is1d)
    {
        shaderParam.global_scale[0]  = 1;
        shaderParam.global_scale[1]  = 1;
        shaderParam.global_scale[2]  = 1;
    }
    else if (axis == 1)
    {
        shaderParam.global_scale[0]  = 1;
        shaderParam.global_scale[1]  = 8;
        shaderParam.global_scale[2]  = 1;
    }
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = (height + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1];
    shaderParam.global_size[2]   = chn;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniExtract1to8Short_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x00000000, 0x00000000, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract1to8Short_2x8", &uniExtract1to8Short_2x8);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }

    return status;
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel_preprocess,
    vsi_nn_kernel_t* kernel,
    int32_t axis
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input1_dtype = I32;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int32_t is1d = inputs[0]->attr.dim_num == 1 ? 1 : 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == F16)
    {
        input0_dtype = I16;
    }
    if (output_dtype == F16)
    {
        output_dtype = I16;
    }

    if (input0_dtype == I8)
    {
        input0_dtype = U8;
    }
    if (output_dtype == I8)
    {
        output_dtype = U8;
    }

    key = HASH_PREPROCESS_KEY( input1_dtype, I32 );

    for( i = 0; i < _cnt_of_array(_preprocess_kernel_map); i ++ )
    {
        if ( _preprocess_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_preprocess_kernel_map) )
    {
        snprintf( kernel_preprocess->info.name, VX_MAX_KERNEL_NAME, "%s",  _preprocess_kernel_map[i].function_name );
        kernel_preprocess->info.parameters = _preprocess_kernel_param_def;
        kernel_preprocess->info.numParams = _REPEAT_PREPROCESS_PARAM_NUM;
        kernel_preprocess->info.initialize = _preprocess_initializer;

        vsi_nn_kernel_add_source( kernel_preprocess, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _preprocess_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_preprocess, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _preprocess_kernel_map[i].source_name );
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

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _repeat_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t preprocess_node_params[_REPEAT_PREPROCESS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t node_params[_REPEAT_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t tmp_node = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * kernel_preprocess = NULL;
    vsi_nn_tensor_t * tensor_preprocess = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_input1 = NULL, rs_output = NULL;
    int32_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }};
    int32_t new_rank[2] = {0, 0};
    int32_t axis  = vsi_nn_kernel_param_get_int32( params, "axis" );

    // Check if gpu can support the size
    if ( !vsi_nn_kernel_gpu_check_shape(
        (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num ) )
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
    }

    if (inputs[1]->attr.dim_num == 1)
    {
        new_shape[0][0] = inputs[1]->attr.size[0];
        new_shape[0][1] = 1;
        rs_input1 = vsi_nn_kernel_tensor_reshape(inputs[1]->t, new_shape[0], 2);
    }

    kernel_preprocess = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
    // Assign unique_id
    kernel_preprocess->unique_id = kernel->unique_id;

    status = _query_kernel( inputs, outputs, kernel_preprocess, kernel, axis );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = inputs[1]->attr.size[0];
    attr.size[1] = 1;
    attr.dim_num = 2;
    tensor_preprocess = vsi_nn_CreateTensor( graph, &attr );

    // preprocess
    tmp_node = vsi_nn_kernel_create_node( graph, kernel_preprocess );
    if (tmp_node)
    {
        uint32_t index = 0;
        if (rs_input1)
        {
            preprocess_node_params[index++] = rs_input1;
        }
        else
        {
            preprocess_node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
        }
        preprocess_node_params[index++] = (vsi_nn_kernel_node_param_t)tensor_preprocess->t;

        status  = vsi_nn_kernel_node_pass_param( tmp_node, preprocess_node_params,
            _REPEAT_PREPROCESS_PARAM_NUM );
        CHECK_STATUS(status);
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U8 = 0;
            border.constant_value.U16 = 0;
            border.constant_value.S32 = 0;
            if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
            {
                border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
            }
            status = vxSetNodeAttribute( (vx_node)tmp_node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }

    // repeat
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
        node_params[index++] = (vsi_nn_kernel_node_param_t)tensor_preprocess->t;
        if (rs_output)
        {
            node_params[index++] = rs_output;
        }
        else
        {
            node_params[index++] = (vsi_nn_kernel_node_param_t)outputs[0]->t;
        }
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis );

        status  = vsi_nn_kernel_node_pass_param( node, node_params,
            _REPEAT_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &node_params[4] );
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_REPLICATE;
            status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
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
    if ( kernel_preprocess )
    {
        vsi_nn_kernel_release( &kernel_preprocess );
    }
    if ( tensor_preprocess )
    {
        vsi_nn_ReleaseTensor( &tensor_preprocess );
    }
    if (tmp_node) {vsi_nn_kernel_node_release( &tmp_node );}
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( repeat, _setup )


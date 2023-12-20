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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

#define KERNEL_SOURCE_0    "maxpool",

 typedef enum
{
    _error = -1,
    _MAX = 0,
    _AVG
} vsi_nn_pool_type_e;

#define HASH_POOL_KEY(_input_type, _output_type, _pool_type, _image_2d) \
    ((_input_type << 24) | (_output_type << 16) | (_pool_type << 8) | (_image_2d))

#define HASH_MAXPOOL_SH_KERNEL_NAME(SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.maxpool_"#SRC_TYPE"to"#DST_TYPE)

#define MAXPOOL_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_POOL_KEY(IN0_TYPE, OUT_TYPE, _MAX, 0), \
        HASH_MAXPOOL_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    MAXPOOL_KERNELS(F16,  F16,        KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(BF16, BF16,       KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(I8,   I8,         KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(U8,   U8,         KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(I16,  I16,        KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(U8,   F16,        KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(I8,   F16,        KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(I16,  F16,        KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(F16,  I8,         KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(F16,  U8,         KERNEL_SOURCE_0)
    MAXPOOL_KERNELS(F16,  I16,        KERNEL_SOURCE_0)
};

static vx_param_description_t kernel_param_def[] =
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
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

DEF_KERNEL_INITIALIZER(_maxpool_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
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
    float input_zp     = 0.0f;
    float input_scale  = 1.0f;
    float output_zp    = 0;
    float output_scale = 1.0f;
    float inout_scale  = 1.0f;
    float inout_tail   = 0.0f;
    int32_t width      = 0;
    int32_t height     = 0;

    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * out_shape = NULL;
    uint32_t pack_key = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_shape  = attr[1]->shape;

    width = (int32_t)attr[0]->shape->data[0];
    height = (int32_t)attr[0]->shape->data[1];

    input_scale  = attr[0]->scale;
    input_zp     = (float)attr[0]->zero_point;
    output_scale = attr[1]->scale;
    output_zp    = (float)attr[1]->zero_point;

    inout_scale = input_scale / output_scale;
    inout_tail = output_zp - input_zp * inout_scale;

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE )    \
        (IN0_TYPE | ( OUT_TYPE << 16))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype );

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;

    gpu_param.global_size[0] = (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0];
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    {
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvF16toFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node, "inout_scale", &inout_scale );
        status |= vsi_nn_kernel_gpu_add_param( node, "inout_tail", &inout_tail );
        status |= vsi_nn_kernel_gpu_add_param( node, "width", &width );
        status |= vsi_nn_kernel_gpu_add_param( node, "height", &height );
        CHECK_STATUS_FAIL_GOTO(status, final);

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( I8,  I8  ):
        case _PACK_SELECT_KEY( U8,  U8  ):
        case _PACK_SELECT_KEY( I16, I16 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8",  &uniConvertInt32toUint8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        case _PACK_SELECT_KEY( F16, I8  ):
        case _PACK_SELECT_KEY( F16, U8  ):
        case _PACK_SELECT_KEY( F16, I16 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvF16toFp32_4x4",  &uniConvF16toFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8",  &uniConvertInt32toUint8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8",  &uniConvBF16toF32_Part0_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractOddData_2x8",  &uniExtractOddData_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        default:
            break;
        }
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

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
} /* _maxpool_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t pool_type,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_POOL_KEY( input0_dtype, output_dtype, pool_type, 0 );

    for ( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _maxpool_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t pool_type         = vsi_nn_kernel_param_get_int32( params, "pool_type" );
    int32_t pool_size_x       = vsi_nn_kernel_param_get_int32( params, "pool_size_x" );
    int32_t pool_size_y       = vsi_nn_kernel_param_get_int32( params, "pool_size_y" );
    int32_t pool_pad_x_left   = vsi_nn_kernel_param_get_int32( params, "pool_pad_x_left" );
    int32_t pool_pad_y_top    = vsi_nn_kernel_param_get_int32( params, "pool_pad_y_top" );
    int32_t stride_x          = vsi_nn_kernel_param_get_int32( params, "stride_x" );
    int32_t stride_y          = vsi_nn_kernel_param_get_int32( params, "stride_y" );
    int32_t dilation_x        = vsi_nn_kernel_param_get_int32( params, "dilation_x" );
    int32_t dilation_y        = vsi_nn_kernel_param_get_int32( params, "dilation_y" );
    int32_t kernel_dia_x      = pool_size_x * dilation_x;
    int32_t kernel_dia_y      = pool_size_y * dilation_y;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(params);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, pool_type, kernel );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 2;
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_PARAM_NUM,
                    inputs, 1, outputs, 1 );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_x );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_y );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pool_pad_x_left );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pool_pad_y_top );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_dia_x );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_dia_y );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_x );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_y );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_scalar_release( &tmp_params[5] );
            vsi_nn_kernel_scalar_release( &tmp_params[6] );
            vsi_nn_kernel_scalar_release( &tmp_params[7] );
            vsi_nn_kernel_scalar_release( &tmp_params[8] );
            vsi_nn_kernel_scalar_release( &tmp_params[9] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( pool, _setup )

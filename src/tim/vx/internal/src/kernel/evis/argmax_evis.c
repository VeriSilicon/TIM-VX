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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

#define HASH_ARGMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, _image_2d) \
    ((AXIS << 20) | (IN_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

 #define HASH_ARGMAX_KERNEL_SOURCE_NAME(AXIS) \
    "argmax_axis"#AXIS

#define HASH_ARGMAX_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_ARGMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0), \
        CVIVANTE_NAMESPACE("evis.argmax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE), \
        HASH_ARGMAX_KERNEL_SOURCE_NAME(AXIS) },

#define HASH_ARGMAX_KERNELS_2D( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_ARGMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("evis.argmax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE"_2D"), \
        HASH_ARGMAX_KERNEL_SOURCE_NAME(AXIS) },

#define HASH_ARGMAX_KERNELS_HALF( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_ARGMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0), \
        CVIVANTE_NAMESPACE("evis.argmax_axis"#AXIS"_F16to"#OUT_DTYPE), \
        HASH_ARGMAX_KERNEL_SOURCE_NAME(AXIS) },

#define HASH_ARGMAX_KERNELS_HALF_2D( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_ARGMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("evis.argmax_axis"#AXIS"_F16to"#OUT_DTYPE"_2D"), \
        HASH_ARGMAX_KERNEL_SOURCE_NAME(AXIS) },

#define HASH_ARGMAX_KERNELS_MIX_OPT( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_ARGMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 2), \
        CVIVANTE_NAMESPACE("evis.argmax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE"_opt"), \
        HASH_ARGMAX_KERNEL_SOURCE_NAME(AXIS) },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } _argmax_evis_kernel_map[] =
{
    HASH_ARGMAX_KERNELS_HALF(0, F16,  U8)
    HASH_ARGMAX_KERNELS_HALF(0, F16,  I16)
    HASH_ARGMAX_KERNELS_HALF(0, BF16, U8)
    HASH_ARGMAX_KERNELS_HALF(0, BF16, I16)
    HASH_ARGMAX_KERNELS_HALF(1, F16,  U8)
    HASH_ARGMAX_KERNELS_HALF(1, F16,  I16)
    HASH_ARGMAX_KERNELS_HALF(1, BF16, U8)
    HASH_ARGMAX_KERNELS_HALF(1, BF16, I16)
    HASH_ARGMAX_KERNELS_HALF(2, F16,  U8)
    HASH_ARGMAX_KERNELS_HALF(2, F16,  I16)
    HASH_ARGMAX_KERNELS_HALF(2, BF16, U8)
    HASH_ARGMAX_KERNELS_HALF(2, BF16, I16)

    HASH_ARGMAX_KERNELS_HALF_2D(0, F16,  U8)
    HASH_ARGMAX_KERNELS_HALF_2D(0, F16,  I16)
    HASH_ARGMAX_KERNELS_HALF_2D(0, BF16, U8)
    HASH_ARGMAX_KERNELS_HALF_2D(0, BF16, I16)
    HASH_ARGMAX_KERNELS_HALF_2D(1, F16,  U8)
    HASH_ARGMAX_KERNELS_HALF_2D(1, F16,  I16)
    HASH_ARGMAX_KERNELS_HALF_2D(1, BF16, U8)
    HASH_ARGMAX_KERNELS_HALF_2D(1, BF16, I16)
    HASH_ARGMAX_KERNELS_HALF_2D(2, F16,  U8)
    HASH_ARGMAX_KERNELS_HALF_2D(2, F16,  I16)
    HASH_ARGMAX_KERNELS_HALF_2D(2, BF16, U8)
    HASH_ARGMAX_KERNELS_HALF_2D(2, BF16, I16)

    HASH_ARGMAX_KERNELS(0, I8,  U8)
    HASH_ARGMAX_KERNELS(0, I8,  I16)
    HASH_ARGMAX_KERNELS(0, U8,  U8)
    HASH_ARGMAX_KERNELS(0, U8,  I16)
    HASH_ARGMAX_KERNELS(0, I16, U8)
    HASH_ARGMAX_KERNELS(0, I16, I16)
    HASH_ARGMAX_KERNELS(1, I8,  U8)
    HASH_ARGMAX_KERNELS(1, I8,  I16)
    HASH_ARGMAX_KERNELS(1, U8,  U8)
    HASH_ARGMAX_KERNELS(1, U8,  I16)
    HASH_ARGMAX_KERNELS(1, I16, U8)
    HASH_ARGMAX_KERNELS(1, I16, I16)
    HASH_ARGMAX_KERNELS(2, I8,  U8)
    HASH_ARGMAX_KERNELS(2, I8,  I16)
    HASH_ARGMAX_KERNELS(2, U8,  U8)
    HASH_ARGMAX_KERNELS(2, U8,  I16)
    HASH_ARGMAX_KERNELS(2, I16, U8)
    HASH_ARGMAX_KERNELS(2, I16, I16)

    HASH_ARGMAX_KERNELS_2D(0, I8,  U8)
    HASH_ARGMAX_KERNELS_2D(0, I8,  I16)
    HASH_ARGMAX_KERNELS_2D(0, U8,  U8)
    HASH_ARGMAX_KERNELS_2D(0, U8,  I16)
    HASH_ARGMAX_KERNELS_2D(0, I16, U8)
    HASH_ARGMAX_KERNELS_2D(0, I16, I16)
    HASH_ARGMAX_KERNELS_2D(1, I8,  U8)
    HASH_ARGMAX_KERNELS_2D(1, I8,  I16)
    HASH_ARGMAX_KERNELS_2D(1, U8,  U8)
    HASH_ARGMAX_KERNELS_2D(1, U8,  I16)
    HASH_ARGMAX_KERNELS_2D(1, I16, U8)
    HASH_ARGMAX_KERNELS_2D(1, I16, I16)
    HASH_ARGMAX_KERNELS_2D(2, I8,  U8)
    HASH_ARGMAX_KERNELS_2D(2, I8,  I16)
    HASH_ARGMAX_KERNELS_2D(2, U8,  U8)
    HASH_ARGMAX_KERNELS_2D(2, U8,  I16)
    HASH_ARGMAX_KERNELS_2D(2, I16, U8)
    HASH_ARGMAX_KERNELS_2D(2, I16, I16)
    HASH_ARGMAX_KERNELS_MIX_OPT(2, U8,  I16)
    HASH_ARGMAX_KERNELS_MIX_OPT(2, I8,  I16)
};

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

#define SCALAR_INPUT_AXIS          (2)

DEF_KERNEL_INITIALIZER(_argmax_initializer)
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
    int32_t     axis                        = 0;
    uint32_t    argLenSub1                  = 0;
    vsi_nn_kernel_tensor_attr_t * attr[2]   = { NULL, NULL };
    vsi_size_array_t * input_shape           = NULL;
    vsi_size_array_t * output_shape          = NULL;
    uint32_t    packedArgIdx[4]             = {0};

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input_shape   = attr[0]->shape;
    output_shape  = attr[1]->shape;

    if (axis == 2 && input_shape->data[2] == 1)
    {
        argLenSub1 = (uint32_t)(input_shape->data[1] - 1);
    }
    else
    {
        if (axis == 2)
            argLenSub1 = (uint32_t)(input_shape->data[2] - 1);
        else if (axis == 1)
            argLenSub1 = (uint32_t)(input_shape->data[1] - 1);
    }

    if (axis == 0)
    {
        gpu_param.global_scale[0]  = 1;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;

        if (attr[0]->dtype == F16 || attr[0]->dtype == BF16)
        {
            packedArgIdx[0] = 0x00000000;
            packedArgIdx[1] = 0x00000001;
            packedArgIdx[2] = 0x00000002;
            packedArgIdx[3] = 0x00000003;
        }
        else if (attr[1]->dtype == I8 || attr[1]->dtype == U8)
        {
            packedArgIdx[0] = 0x03020100;
            packedArgIdx[1] = 0x07060504;
            packedArgIdx[2] = 0x0b0a0908;
            packedArgIdx[3] = 0x0f0e0d0c;
        }
        else
        {
            packedArgIdx[0] = 0x00010000;
            packedArgIdx[1] = 0x00030002;
            packedArgIdx[2] = 0x00050004;
            packedArgIdx[3] = 0x00070006;
        }
    }
    else
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
        packedArgIdx[0] = packedArgIdx[1] = (argLenSub1 << 16) | (argLenSub1 & 0xFFFF);
        packedArgIdx[2] = packedArgIdx[3] = (argLenSub1 << 16) | (argLenSub1 & 0xFFFF);

        if (attr[0]->dtype == I8 ||
            attr[0]->dtype == U8)
        {
            if (axis == 2 &&
                input_shape->data[2] > 1 &&
                ((attr[1]->dtype == I8 || attr[1]->dtype == U8)
                  || (attr[1]->dtype == I16 && input_shape->data[2] < 256)))
            {
                uint32_t pack = ((argLenSub1 & 0xFF) << 24) | ((argLenSub1 & 0xFF) << 16)
                                 | ((argLenSub1 & 0xFF) << 8) | (argLenSub1 & 0xFF);
                packedArgIdx[0] = packedArgIdx[1] = pack;
                packedArgIdx[2] = packedArgIdx[3] = pack;
                gpu_param.global_scale[0]  = 16;
            }
            else if ( attr[1]->dtype == I8 ||
                 attr[1]->dtype == U8)
            {
                uint32_t pack = ((argLenSub1 & 0xFF) << 24) | ((argLenSub1 & 0xFF) << 16)
                                 | ((argLenSub1 & 0xFF) << 8) | (argLenSub1 & 0xFF);
                packedArgIdx[0] = packedArgIdx[1] = pack;
                packedArgIdx[2] = packedArgIdx[3] = pack;
            }
        }
    }

    gpu_param.global_size[0] = gpu_align_p2(
            (output_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (output_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = output_shape->size > 2 ? output_shape->data[2] : 1;

    switch( axis )
    {
        case 0:
        {
            gpu_dp_inst_t uniPackedIdxAddSat_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0xffff0001, 0xffff0001, 0xffff0001, 0xffff0001,
                0xffff0001, 0xffff0001, 0xffff0001, 0xffff0001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniSrcT2DstT_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff,
                0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConvertHalf2Float32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            if( attr[0]->dtype == F16 || attr[0]->dtype == BF16)
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertHalf2Float32_4x4", &uniConvertHalf2Float32_4x4 );
            }
            else
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniPackedIdxAddSat_2x8", &uniPackedIdxAddSat_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniSrcT2DstT_2x8", &uniSrcT2DstT_2x8 );
            }
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "inputWidth", &input_shape->data[0] );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "packedArgIdx", packedArgIdx );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case 1:
        {
            gpu_dp_inst_t uniExtractData_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractData_2x8", &uniExtractData_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "argLenSub1", &argLenSub1 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "packedArgIdx", packedArgIdx );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case 2:
        {
            gpu_dp_inst_t uniExtractData_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtract1stU8toI16_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtract2ndU8toI16_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractData_2x8", &uniExtractData_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtract1stU8toI16_2x8", &uniExtract1stU8toI16_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtract2ndU8toI16_2x8", &uniExtract2ndU8toI16_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "argLenSub1", &argLenSub1 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "packedArgIdx", packedArgIdx );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        break;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0]) vsi_nn_kernel_tensor_attr_release( &attr[0] );
    if (attr[1]) vsi_nn_kernel_tensor_attr_release( &attr[1] );

    return status;
} /* _argmax_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t axis,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    size_t i;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if ((input_dtype == I8 || input_dtype == U8)
        && output_dtype == I16
        && axis == 2
        && inputs[0]->attr.size[2] < 256
        && image_2d == 0)
    {
        image_2d = 2;
    }

    key = HASH_ARGMAX_HASH_KEY( axis, input_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(_argmax_evis_kernel_map); i ++ )
    {
        if( _argmax_evis_kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(_argmax_evis_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _argmax_evis_kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _argmax_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _argmax_evis_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _argmax_evis_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_EVIS_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    axis = vsi_nn_kernel_param_get_int32(params, "axis");

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num )
     || axis > 2)
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( inputs, outputs, axis, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _EVIS_PARAM_NUM,
                    inputs, 1, outputs, 1 );
            node_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(
                    graph, I32, &axis );

            status = vsi_nn_kernel_node_pass_param( node, node_params, _EVIS_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( argmax, _setup )


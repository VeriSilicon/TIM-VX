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

#if !(VX_TENSOR_POW_API_SUPPORT)
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

#define KERNEL_SOURCE    "pow",

#define HASH_POW_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define TENSOR_POW_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { HASH_POW_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        CVIVANTE_NAMESPACE("evis.pow_"#IN0_TYPE"_"#IN1_TYPE"to"#OUT_TYPE), \
        KERNEL_SOURCE },

#define TENSOR_POW_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { HASH_POW_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        CVIVANTE_NAMESPACE("evis.pow_"#IN0_TYPE"_"#IN1_TYPE"to"#OUT_TYPE"_2D"), \
        KERNEL_SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } pow_map[] =
{
    TENSOR_POW_KERNELS(F16, F16, F16)
    TENSOR_POW_KERNELS(F16, F16, U8)
    TENSOR_POW_KERNELS(F16, U8, F16)
    TENSOR_POW_KERNELS(F16, U8, U8)

    TENSOR_POW_KERNELS(F16, F16, I8)
    TENSOR_POW_KERNELS(F16, I8, F16)
    TENSOR_POW_KERNELS(F16, I8, I8)

    TENSOR_POW_KERNELS(F16, F16, I16)
    TENSOR_POW_KERNELS(F16, I16, F16)
    TENSOR_POW_KERNELS(F16, I16, I16)

    TENSOR_POW_KERNELS(U8, F16, F16)
    TENSOR_POW_KERNELS(U8, F16, U8)
    TENSOR_POW_KERNELS(U8, U8, U8)
    TENSOR_POW_KERNELS(U8, U8, F16)

    TENSOR_POW_KERNELS(I8, F16, F16)
    TENSOR_POW_KERNELS(I8, F16, I8)
    TENSOR_POW_KERNELS(I8, I8, I8)

    TENSOR_POW_KERNELS(I16, F16, F16)
    TENSOR_POW_KERNELS(I16, F16, I16)
    TENSOR_POW_KERNELS(I16, I16, I16)
    TENSOR_POW_KERNELS(BF16, BF16, BF16)

    TENSOR_POW_KERNELS_2D(F16, F16, F16)
    TENSOR_POW_KERNELS_2D(F16, F16, U8)
    TENSOR_POW_KERNELS_2D(F16, U8, F16)
    TENSOR_POW_KERNELS_2D(F16, U8, U8)

    TENSOR_POW_KERNELS_2D(F16, F16, I8)
    TENSOR_POW_KERNELS_2D(F16, I8, F16)
    TENSOR_POW_KERNELS_2D(F16, I8, I8)

    TENSOR_POW_KERNELS_2D(F16, F16, I16)
    TENSOR_POW_KERNELS_2D(F16, I16, F16)
    TENSOR_POW_KERNELS_2D(F16, I16, I16)

    TENSOR_POW_KERNELS_2D(U8, F16, F16)
    TENSOR_POW_KERNELS_2D(U8, F16, U8)
    TENSOR_POW_KERNELS_2D(U8, U8, U8)
    TENSOR_POW_KERNELS_2D(U8, U8, F16)

    TENSOR_POW_KERNELS_2D(I8, F16, F16)
    TENSOR_POW_KERNELS_2D(I8, F16, I8)
    TENSOR_POW_KERNELS_2D(I8, I8, I8)

    TENSOR_POW_KERNELS_2D(I16, F16, F16)
    TENSOR_POW_KERNELS_2D(I16, F16, I16)
    TENSOR_POW_KERNELS_2D(I16, I16, I16)
    TENSOR_POW_KERNELS_2D(BF16, BF16, BF16)
};

static vx_param_description_t vxPowKernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#define _EVIS_POW_PARAM_NUM          _cnt_of_array(vxPowKernel_param_def)

DEF_KERNEL_INITIALIZER(_pow_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float    input0_scale = 1.0f;
    float    input1_scale = 1.0f;
    float    input0_tail = 0;
    float    input1_tail = 0;
    float    output_scale = 1.0f;
    float    output_zp = 0;

    uint32_t pack_key      = 0;
    // dim number ???
    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    out_shape    = attr[2]->shape;
    input0_scale = attr[0]->scale;
    input0_tail  = 0 - (float)attr[0]->zero_point * input0_scale;
    input1_scale = attr[1]->scale;
    input1_tail  = 0 - (float)attr[1]->zero_point * input1_scale;
    output_zp    = (float)attr[2]->zero_point;
    output_scale = 1.0f / attr[2]->scale;

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 8) | ( OUT_TYPE << 16))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype,
            attr[1]->dtype, attr[2]->dtype );

    shaderParam.global_scale[0]  = 8;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((out_shape->data[0] + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = gpu_align_p2((out_shape->data[1] + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1], 2);
    shaderParam.global_size[2]   = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    switch( pack_key )
    {
    case _PACK_SELECT_KEY( BF16, BF16, BF16 ):
        {
            gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x01050004, 0x03070206, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x05050404, 0x07070606, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtractOddData_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x07050301, 0x07050301, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8",
                &uniConvBF16toF32_Part0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8",
                &uniConvBF16toF32_Part1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                &uniExtractOddData_2x8);
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }
        break;
    default:
        {
            gpu_dp_inst_t uniConvertFstDataToFp32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConvertSecDataToFp32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtact8Bit_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniExtactHalf8_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16};

            status = vsi_nn_kernel_gpu_add_param(node, "uniConvertFstDataToFp32_4x4",
                &uniConvertFstDataToFp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecDataToFp32_4x4",
                &uniConvertSecDataToFp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "input0_scale", &input0_scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "input1_scale", &input1_scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "input0_tail", &input0_tail);
            status |= vsi_nn_kernel_gpu_add_param( node, "input1_tail", &input1_tail);
            status |= vsi_nn_kernel_gpu_add_param( node, "output_scale", &output_scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "output_zp", &output_zp);
            if (attr[2]->dtype == F16)
            {
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtact8Bit_2x8",
                    &uniExtactHalf8_2x8);
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param(node, "uniExtact8Bit_2x8",
                    &uniExtact8Bit_2x8);
            }
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }
        break;
    }
#undef _PACK_SELECT_KEY

OnError:
    if ( attr[0] )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if ( attr[1] )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if ( attr[2] )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _pow_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype;
    vsi_nn_kernel_dtype_e input1_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_POW_KEY( input0_dtype, input1_dtype, output_dtype, image_2d );

    for ( i = 0; i < _cnt_of_array(pow_map); i ++ )
    {
        if ( pow_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(pow_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  pow_map[i].function_name );
        kernel->info.parameters = vxPowKernel_param_def;
        kernel->info.numParams = _cnt_of_array( vxPowKernel_param_def );
        kernel->info.initialize = _pow_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                pow_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                pow_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_POW_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(params);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2);
    status = _query_kernel( inputs, outputs, image_2d, kernel );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_POW_PARAM_NUM,
                inputs, 2, outputs, 1 );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_POW_PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( pow, _setup )
#endif

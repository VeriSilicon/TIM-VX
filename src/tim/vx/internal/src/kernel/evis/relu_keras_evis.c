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

#define _RELU_KERAS_KERNEL_SOURCE      "relu_keras"

#define STR(a) #a
// Add kernel hashtable here
#define RELU_KERAS_HASH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (_image_2d))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { RELU_KERAS_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0 ), \
          CVIVANTE_NAMESPACE("evis.relu_keras_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_3D"), \
          _RELU_KERAS_KERNEL_SOURCE }

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE ) \
        { RELU_KERAS_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1 ), \
          CVIVANTE_NAMESPACE("evis.relu_keras_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
          _RELU_KERAS_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _relu_keras_kernel_map[] =
{
    PACK_KERNEL_MAP(BF16, BF16),
    PACK_KERNEL_MAP(F16,  F16),
    PACK_KERNEL_MAP(F16,  I16),
    PACK_KERNEL_MAP(F16,  I8),
    PACK_KERNEL_MAP(F16,  U8),
    PACK_KERNEL_MAP(I16,  I16),
    PACK_KERNEL_MAP(I16,  F16),
    PACK_KERNEL_MAP(I8,   I8),
    PACK_KERNEL_MAP(I8,   F16),
    PACK_KERNEL_MAP(U8,   U8),
    PACK_KERNEL_MAP(U8,   F16),
    PACK_KERNEL_MAP_2D(BF16, BF16),
    PACK_KERNEL_MAP_2D(F16,  F16),
    PACK_KERNEL_MAP_2D(F16,  I16),
    PACK_KERNEL_MAP_2D(F16,  I8),
    PACK_KERNEL_MAP_2D(F16,  U8),
    PACK_KERNEL_MAP_2D(I16,  I16),
    PACK_KERNEL_MAP_2D(I16,  F16),
    PACK_KERNEL_MAP_2D(I8,   I8),
    PACK_KERNEL_MAP_2D(I8,   F16),
    PACK_KERNEL_MAP_2D(U8,   U8),
    PACK_KERNEL_MAP_2D(U8,   F16),
};


/*
 * Kernel params
 */
static vx_param_description_t _relu_keras_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _RELU_KERAS_PARAM_NUM  _cnt_of_array( _relu_keras_kernel_param_def )

#define SCALAR_ALPHA              (2)
#define SCALAR_MAX_VALUE          (3)
#define SCALAR_THRESHOLD          (4)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_relu_keras_initializer)
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
    vsi_nn_kernel_tensor_attr_t * output_attr    = NULL;
    vsi_nn_kernel_tensor_attr_t * input_attr     = NULL;
    vsi_size_array_t             * out_shape      = NULL;
    vsi_nn_kernel_dtype_e         input_dtype    = F16;
    vsi_nn_kernel_dtype_e         output_dtype   = F16;
    float                         alpha          = 0.0f;
    float                         threshold      = 0.0f;
    float                         offset         = 0.0f;
    float                         scaleIn        = 1.0f;
    float                         scaleOut       = 1.0f;
    float                         inputTail      = 0.0f;
    float                         output_ZP      = 0;
    float                         input_ZP       = 0;
    int32_t                       srcFixPointPos = 0;
    int32_t                       dstFixPointPos = 0;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );
    out_shape  = output_attr->shape;
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_ALPHA], &(alpha));
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_THRESHOLD], &(threshold));
    input_dtype  = input_attr->dtype;
    output_dtype = output_attr->dtype;
    offset       = alpha * threshold;

    if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        srcFixPointPos   = input_attr->dfp.fl;
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant)
    {
        input_ZP         = (float)(input_attr->asymm.zero_point);
        scaleIn          = input_attr->asymm.scale;
    }

    if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        dstFixPointPos   = output_attr->dfp.fl;
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant)
    {
        output_ZP        = (float)(output_attr->asymm.zero_point);
        scaleOut         = 1.0f / output_attr->asymm.scale;
    }

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.dim = out_shape->size < 3 ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    if (VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant)
    {
        inputTail = -input_ZP * scaleIn;
        status  = vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
        status |= vsi_nn_kernel_gpu_add_param(node, "inputTail", &inputTail);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        if (srcFixPointPos >=0 )
            scaleIn = 1.0f / (float) ((int64_t)1 << srcFixPointPos);
        else
            scaleIn = (float) ((int64_t)1 << -srcFixPointPos);

        status = vsi_nn_kernel_gpu_add_param(node, "input_scale", &scaleIn);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if (VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant)
    {
        status  = vsi_nn_kernel_gpu_add_param(node, "output_scale", &scaleOut);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputZP", &output_ZP);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        if (dstFixPointPos >=0 )
            scaleOut = (float) ((int64_t)1 << dstFixPointPos);
        else
            scaleOut = 1.0f / (float) ((int64_t)1 << -dstFixPointPos);

        status  = vsi_nn_kernel_gpu_add_param(node, "output_scale", &scaleOut);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if (F16 == input_dtype)
    {
        gpu_dp_inst_t uniConvIntegertoFP32_Lo_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvIntegertoFP32_Hi_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param(node, "uniConvFP16toFP32_Lo_4x4", &uniConvIntegertoFP32_Lo_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvFP16toFP32_Hi_4x4", &uniConvIntegertoFP32_Hi_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (BF16 == input_dtype)
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
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        gpu_dp_inst_t uniConvIntegertoFP32_Lo_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvIntegertoFP32_Hi_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param(node, "uniConvIntegertoFP32_Lo_4x4", &uniConvIntegertoFP32_Lo_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvIntegertoFP32_Hi_4x4", &uniConvIntegertoFP32_Hi_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if (F16 == output_dtype)
    {
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param(node, "uniExtractHalf8_2x8", &uniExtractHalf8_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (BF16 == output_dtype)
    {
        gpu_dp_inst_t uniPackedBF16_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param(node, "uniPackedBF16_2x8", &uniPackedBF16_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        gpu_dp_inst_t uniExtractInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param(node, "uniExtractInteger_2x8", &uniExtractInteger_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_add_param(node, "offset", &offset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    SAFE_FREE_TENSOR_ATTR(input_attr);
    return status;

} /* _relu_keras_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _relu_keras_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _relu_keras_kernel_map );
    vx_param_description_t * param_def  = _relu_keras_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _relu_keras_kernel_param_def );
    vx_kernel_initialize_f  initializer = _relu_keras_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = RELU_KERAS_HASH_KEY( in_dtype, out_dtype, image_2d );

    for( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < (uint32_t)kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
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
    vsi_nn_kernel_node_param_t node_params[_RELU_KERAS_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    float   alpha      = vsi_nn_kernel_param_get_float32( params, "alpha" );
    float   max_value  = vsi_nn_kernel_param_get_float32( params, "max_value" );
    float   threshold  = vsi_nn_kernel_param_get_float32( params, "threshold" );

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status   = _query_kernel( kernel, inputs, outputs, image_2d );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _RELU_KERAS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_ALPHA]     = vsi_nn_kernel_scalar_create( graph, F32, &alpha );
            node_params[SCALAR_MAX_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &max_value );
            node_params[SCALAR_THRESHOLD] = vsi_nn_kernel_scalar_create( graph, F32, &threshold );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _RELU_KERAS_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_ALPHA] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MAX_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_THRESHOLD] );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( relu_keras, _setup )


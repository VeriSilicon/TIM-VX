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
typedef enum
{
    LARGE = 0,
    SMALL
} _internal_nearest_e;

#define _RESIZE_1D_NEAREST_KERNEL_SOURCE      "resize_1d_nearest"

#define STR(a) #a
// Add kernel hashtable here
#define RESIZE_1D_NEAREST_HASH_KEY( IN_DTYPE, OUT_DTYPE, mode ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (mode))


#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_1D_NEAREST_HASH_KEY( IN_DTYPE, OUT_DTYPE, LARGE ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_nearest_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
          _RESIZE_1D_NEAREST_KERNEL_SOURCE }

#define PACK_KERNEL_MAP_OPT( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_1D_NEAREST_HASH_KEY( IN_DTYPE, OUT_DTYPE, SMALL ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_nearest_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_op"), \
          _RESIZE_1D_NEAREST_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _resize_1d_nearest_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP(F16, F16),
    PACK_KERNEL_MAP(I16, I16),
    PACK_KERNEL_MAP(I8, I8),
    PACK_KERNEL_MAP(U8, U8),
    PACK_KERNEL_MAP_OPT(F16, F16),
    PACK_KERNEL_MAP_OPT(I16, I16),
    PACK_KERNEL_MAP_OPT(I8, I8),
    PACK_KERNEL_MAP_OPT(U8, U8),
};


/*
 * Kernel params
 */
static vx_param_description_t _resize_1d_nearest_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _RESIZE_1D_NEAREST_PARAM_NUM  _cnt_of_array( _resize_1d_nearest_kernel_param_def )

#define SCALAR_ALIGN_CORNERS         (2)
#define SCALAR_HALF_PIXEL            (3)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_resize_1d_nearest_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
#define MAX_POST_SHIFT_BITS     (31)
#define MAX_MULTIPLIER_NUM      (65535)
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t * input_attr    = NULL;
    vsi_size_array_t             * out_shape     = NULL;
    vsi_size_array_t             * in_shape      = NULL;
    vsi_nn_kernel_dtype_e         input_dtype   = F16;
    vsi_nn_kernel_dtype_e         output_dtype  = F16;
    int32_t     align_corners      = 0;
    int32_t     half_pixel_centers = 0;
    uint32_t    depth              = 0;
    int32_t     srcFixPointPos     = 0;
    int32_t     dstFixPointPos     = 0;
    float       input_scale        = 1.0;
    int32_t     inputZP            = 0;
    float       output_scale       = 1.0;
    int32_t     outputZP           = 0;
    float       scale_factor       = 1.0f;
    uint32_t    in_width           = 0;
    uint32_t    out_width          = 0;
    uint32_t    out_height         = 0;
    float       half_pixel_value   = 0.0f;
    float       round_value        = 0.0f;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &align_corners);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &half_pixel_centers);
    CHECK_STATUS_FAIL_GOTO(status, final );
    out_shape     = output_attr->shape;
    in_shape      = input_attr->shape;
    input_dtype   = input_attr->dtype;
    output_dtype  = output_attr->dtype;

    in_width          = (uint32_t)(in_shape->data[0]);
    depth             = (uint32_t)(in_shape->data[2]);
    out_width         = (uint32_t)(out_shape->data[0]);
    out_height        = (uint32_t)(out_shape->data[1]);

    if (BF16 == input_dtype && output_dtype == BF16)
    {
        input_dtype  = F16;
        output_dtype = F16;
    }
    if (align_corners && out_width > 1)
    {
        scale_factor = ((float)(in_width - 1) * 1.0f) / (float)(out_width - 1);
    }
    else
    {
        scale_factor = ((float)in_width * 1.0f) / (float)out_width;
    }


    if (align_corners)
    {
        round_value = 0.5f;
    }
    else
    {
        round_value = 0.0f;
    }

    if (half_pixel_centers)
    {
        half_pixel_value = 0.5f;
    }
    else
    {
        half_pixel_value = 0.0f;
    }

    if (VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant )
    {
        input_scale    = input_attr->asymm.scale;
        inputZP        = input_attr->asymm.zero_point;
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        srcFixPointPos   = input_attr->dfp.fl;
        if (srcFixPointPos >= 0)
        {
            input_scale = 1.0f / (float) ((int64_t)1 << srcFixPointPos);
        }
        else if (srcFixPointPos < 0)
        {
            input_scale = (float)((int64_t)1 << -srcFixPointPos);
        }
        inputZP = 0;
    }
    else
    {
        input_scale = 1.0f;
        inputZP     = 0;
    }

    if (VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant )
    {
        output_scale   = 1.0f / output_attr->asymm.scale;
        outputZP       = output_attr->asymm.zero_point;
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        dstFixPointPos = output_attr->dfp.fl;
        if (dstFixPointPos >= 0)
        {
            output_scale = (float) ((int64_t)1 << dstFixPointPos);
        }
        else if (dstFixPointPos < 0)
        {
            output_scale = 1.0f / (float) ((int64_t)1 << -dstFixPointPos);
        }
        outputZP = 0;
    }
    else
    {
        output_scale = 1.0;
        outputZP     = 0;
    }

    if (F16 == input_dtype && F16 == output_dtype)
    {
        gpu_dp_inst_t uniGetExtractData_2x8 = {{
            0x00009999, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x00000000, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00100010, 0x00100010, 0x00100010, 0x00100010,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        if (scale_factor < 4.0f)
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniGetExtractData_2x8", &uniGetExtractData_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
        status  = vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if ( input_dtype == output_dtype && (I8 == input_dtype || I16 == input_dtype))
    {
        gpu_dp_inst_t uniGetExtractData_2x8 = {{
            0x00009999, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x00000000, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00080008, 0x00080008, 0x00080008, 0x00080008,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertI8toI8_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        if (I16 == input_dtype)
        {
            uniGetExtractData_2x8.data[8]  = 0x00100010;
            uniGetExtractData_2x8.data[9]  = 0x00100010;
            uniGetExtractData_2x8.data[10] = 0x00100010;
            uniGetExtractData_2x8.data[11] = 0x00100010;
            uniGetExtractData_2x8.data[12] = 0x00100010;
            uniGetExtractData_2x8.data[13] = 0x00100010;
            uniGetExtractData_2x8.data[14] = 0x00100010;
            uniGetExtractData_2x8.data[15] = 0x00100010;
        }

        if (srcFixPointPos > dstFixPointPos)
        {
            int32_t  postshift      = vsi_nn_min(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI8toI8_2x8.data[7] |= (postshift & 0x1F);
        }
        else
        {
            uint32_t multiplier = vsi_nn_min((int64_t)1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            uint32_t i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI8toI8_2x8.data[i + 8] = multiplier;
            }
        }

        if (scale_factor < 4.0f)
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniGetExtractData_2x8", &uniGetExtractData_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;

        status  = vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertI8toI8_2x8", &uniConvertI8toI8_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (U8 == input_dtype && U8 == output_dtype)
    {
        uint16_t  M0                   = 0;
        int32_t   postShift            = 0;
        uint32_t  multAndoutZP[2]      = {0};
        gpu_dp_inst_t uniMultiplyAndPostShift_2x8 = {{
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniGetExtractData_2x8 = {{
            0x00009999, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x00000000, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00080008, 0x00080008, 0x00080008, 0x00080008,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_quantize_multiplier_16bit(input_scale * output_scale, &M0, &postShift);

        multAndoutZP[0] = (uint32_t)(M0);
        multAndoutZP[1] = (uint32_t)((outputZP << postShift) - inputZP * M0);

        uniMultiplyAndPostShift_2x8.data[7] |= (postShift & 0x1F);

        if (scale_factor < 4.0f)
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniGetExtractData_2x8", &uniGetExtractData_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;


        status  = vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP", multAndoutZP);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniMultiplyAndPostShift_2x8", &uniMultiplyAndPostShift_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    status  = vsi_nn_kernel_gpu_add_param( node, "half_pixel_value", &half_pixel_value);
    status |= vsi_nn_kernel_gpu_add_param( node, "round_value", &round_value);
    CHECK_STATUS_FAIL_GOTO(status, final );

    gpu_param.global_size[0]   = gpu_align_p2((out_width  + gpu_param.global_scale[0] - 1)\
                                / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (out_height + gpu_param.global_scale[1] - 1) / gpu_param.global_scale[1];
    gpu_param.global_size[2]   = depth;


    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
#undef MAX_MULTIPLIER_NUM
#undef MAX_POST_SHIFT_BITS
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );
    return status;
} /* _resize_nearest_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
        int32_t align_corners
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype = F16;
    vsi_nn_kernel_dtype_e out_dtype = F16;
    const _kernel_map_type * kernel_map = _resize_1d_nearest_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _resize_1d_nearest_kernel_map );
    vx_param_description_t * param_def  = _resize_1d_nearest_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _resize_1d_nearest_kernel_param_def );
    vx_kernel_initialize_f  initializer = _resize_1d_nearest_initializer;

    uint32_t key = 0;
    uint32_t i   = 0;
    vsi_size_t inputWidth  = inputs[0]->attr.size[0];
    vsi_size_t outputWidth = outputs[0]->attr.size[0];
    float    scale_factor;
    _internal_nearest_e resize_mode = LARGE;

    if (align_corners && outputWidth > 1)
    {
        scale_factor = (vx_float32)(inputWidth - 1) / (vx_float32)(outputWidth - 1);
    }
    else
    {
        scale_factor = (vx_float32)inputWidth / (vx_float32)outputWidth;
    }

    if (scale_factor < 4.0f)
    {
        resize_mode = SMALL;
    }
    else
    {
        resize_mode = LARGE;
    }


    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (BF16 == in_dtype && BF16 == out_dtype)
    {
        in_dtype  = F16;
        out_dtype = F16;
    }

    key = RESIZE_1D_NEAREST_HASH_KEY( in_dtype, out_dtype, resize_mode );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if ( i < (uint32_t)kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_RESIZE_1D_NEAREST_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );

    status = _query_kernel( kernel, inputs, outputs, align_corners );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _RESIZE_1D_NEAREST_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_ALIGN_CORNERS] = vsi_nn_kernel_scalar_create( graph, I32, &align_corners );
            node_params[SCALAR_HALF_PIXEL]    = vsi_nn_kernel_scalar_create( graph, I32, &half_pixel_centers );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _RESIZE_1D_NEAREST_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_ALIGN_CORNERS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_HALF_PIXEL] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( resize_1d_nearest, _setup )


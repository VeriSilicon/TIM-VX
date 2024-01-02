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
#include "utils/vsi_nn_dtype_util_prv.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    DOWN = 0,
    DOWN_2X_SAME,
    DOWN_2X_HALF_SAME,
    UP,
    UP_OPT,
    UP_2X_SAME,
    UP_2X_HALF_SAME,
    UP_4X_SAME,
    UP_4X_HALF_SAME,
    UP_8X_SAME,
    UP_8X_HALF_SAME,
} _internal_scale_e;

#define _RESIZE_BILINEAR_KERNEL_SOURCE(_input_type)      "resize_1d_bilinear_"#_input_type
#define _RESIZE_BILINEAR_KERNEL_SOURCE_OPT(_input_type)  "resize_1d_bilinear_"#_input_type"_opt"
#define _RESIZE_BILINEAR_KERNEL_SOURCE_UP_NX()  "resize_1d_bilinear_UP_NX"
#define _RESIZE_BILINEAR_KERNEL_SOURCE_DOWN_NX()  "resize_1d_bilinear_DOWN_NX"

#define STR(a) #a
// Add kernel hashtable here
#define RESIZE_1D_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, scale_flag ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (scale_flag))

#define PACK_KERNEL_MAP_DOWN( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_1D_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, DOWN ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_DOWN"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_1D_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_UP"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_OPT( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_1D_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_OPT ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_UP_opt"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_OPT(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_NX( IN_DTYPE, OUT_DTYPE, MODE_TYPE ) \
        { RESIZE_1D_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, MODE_TYPE ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_"STR(MODE_TYPE)), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_UP_NX() }

#define PACK_KERNEL_MAP_DOWN_NX( IN_DTYPE, OUT_DTYPE, MODE_TYPE ) \
        { RESIZE_1D_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, MODE_TYPE ), \
          CVIVANTE_NAMESPACE("evis.resize_1d_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_"STR(MODE_TYPE)), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_DOWN_NX() }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _resize_1d_bilinear_kernel_map[] =
{
    PACK_KERNEL_MAP_DOWN(I8, I8),
    PACK_KERNEL_MAP_DOWN(I16, I16),
    PACK_KERNEL_MAP_DOWN(U8, F16),
    PACK_KERNEL_MAP_DOWN(U8, U8),
    PACK_KERNEL_MAP_DOWN(F16, F16),
    PACK_KERNEL_MAP_DOWN(F16, U8),
    PACK_KERNEL_MAP_DOWN(BF16, BF16),
    PACK_KERNEL_MAP_UP(I8, I8),
    PACK_KERNEL_MAP_UP(I16, I16),
    PACK_KERNEL_MAP_UP(U8, U8),
    PACK_KERNEL_MAP_UP(F16, F16),
    PACK_KERNEL_MAP_UP(BF16, BF16),
    PACK_KERNEL_MAP_UP_OPT(U8, U8),
    PACK_KERNEL_MAP_UP_NX(U8, U8, UP_2X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(U8, U8, UP_2X_SAME),
    PACK_KERNEL_MAP_UP_NX(I8, I8, UP_2X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(I8, I8, UP_2X_SAME),
    PACK_KERNEL_MAP_UP_NX(I16, I16, UP_2X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(I16, I16, UP_2X_SAME),
    PACK_KERNEL_MAP_UP_NX(F16, F16, UP_2X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(F16, F16, UP_2X_SAME),
    PACK_KERNEL_MAP_UP_NX(U8, U8, UP_4X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(U8, U8, UP_4X_SAME),
    PACK_KERNEL_MAP_UP_NX(I8, I8, UP_4X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(I8, I8, UP_4X_SAME),
    PACK_KERNEL_MAP_UP_NX(I16, I16, UP_4X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(I16, I16, UP_4X_SAME),
    PACK_KERNEL_MAP_UP_NX(F16, F16, UP_4X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(F16, F16, UP_4X_SAME),
    PACK_KERNEL_MAP_UP_NX(U8, U8, UP_8X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(U8, U8, UP_8X_SAME),
    PACK_KERNEL_MAP_UP_NX(I8, I8, UP_8X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(I8, I8, UP_8X_SAME),
    PACK_KERNEL_MAP_UP_NX(I16, I16, UP_8X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(I16, I16, UP_8X_SAME),
    PACK_KERNEL_MAP_UP_NX(F16, F16, UP_8X_HALF_SAME),
    PACK_KERNEL_MAP_UP_NX(F16, F16, UP_8X_SAME),
    PACK_KERNEL_MAP_DOWN_NX(U8,  U8,  DOWN_2X_HALF_SAME),
    PACK_KERNEL_MAP_DOWN_NX(U8,  U8,  DOWN_2X_SAME),
    PACK_KERNEL_MAP_DOWN_NX(I8,  I8,  DOWN_2X_HALF_SAME),
    PACK_KERNEL_MAP_DOWN_NX(I8,  I8,  DOWN_2X_SAME),
    PACK_KERNEL_MAP_DOWN_NX(I16, I16, DOWN_2X_HALF_SAME),
    PACK_KERNEL_MAP_DOWN_NX(I16, I16, DOWN_2X_SAME),
    PACK_KERNEL_MAP_DOWN_NX(F16, F16, DOWN_2X_HALF_SAME),
    PACK_KERNEL_MAP_DOWN_NX(F16, F16, DOWN_2X_SAME),
};


/*
 * Kernel params
 */
static vx_param_description_t _resize_1d_bilinear_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _RESIZE_1D_BILINEAR_PARAM_NUM  _cnt_of_array( _resize_1d_bilinear_kernel_param_def )
#define _RESIZE_NO_SCALE_PARAM_NUM      4
#define _RESIZE_1D_NX_KERENL_PARAM_NUM  3

#define SCALAR_ALIGN_CORNERS         (2)
#define SCALAR_HALF_PIXEL            (3)
#define SCALAR_TENSOR_SCALE          (4)
#define SCALAR_SCALE_TYPE            (2)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_resize_1d_bilinear_initializer)
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
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t * input_attr    = NULL;
    vsi_size_array_t             * out_shape     = NULL;
    vsi_size_array_t             * in_shape      = NULL;
    vsi_nn_kernel_dtype_e         input_dtype   = F16;
    vsi_nn_kernel_dtype_e         output_dtype  = F16;
    int32_t align_corners = 0;
    int32_t half_pixel_centers = 0;

    uint32_t    depth              = 0;
    float       input_scale        = 1.0;
    int32_t     inputZP            = 0;
    float       output_scale       = 1.0;
    int32_t     outputZP           = 0;
    float       scale_factor       = 1.0f;
    uint32_t    in_width           = 1;
    uint32_t    out_width          = 1;
    uint32_t    out_height         = 1;
    float       half_pixel_value = 0.0f;
    vsi_bool    is_use_scale_kernel = (vsi_bool)(_RESIZE_1D_BILINEAR_PARAM_NUM == param_size);
    _internal_scale_e scale_flag = DOWN;
    vsi_bool    is_run_nx_kernel    = (vsi_bool)(_RESIZE_1D_NX_KERENL_PARAM_NUM == param_size);
    int32_t     scale_type_value = 0;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    if (is_run_nx_kernel)
    {
        status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &scale_type_value);
        CHECK_STATUS_FAIL_GOTO(status, final );
        scale_flag = (_internal_scale_e)scale_type_value;
    }
    else
    {
        status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &align_corners);
        CHECK_STATUS_FAIL_GOTO(status, final );
        status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &half_pixel_centers);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    out_shape     = output_attr->shape;
    in_shape      = input_attr->shape;
    input_dtype   = input_attr->dtype;
    output_dtype  = output_attr->dtype;

    in_width          = (uint32_t)(in_shape->data[0]);
    depth             = (uint32_t)(in_shape->data[2]);
    out_width         = (uint32_t)(out_shape->data[0]);
    out_height        = (uint32_t)(out_shape->data[1]);

    if (align_corners && out_width > 1)
    {
        scale_factor = ((vx_float32)(in_width - 1) * 1.0f) / (vx_float32)(out_width - 1);
    }
    else
    {
        scale_factor = ((vx_float32)in_width * 1.0f) / (vx_float32)out_width;
    }

    if (half_pixel_centers)
    {
        half_pixel_value = 0.5f;
    }
    else
    {
        half_pixel_value = 0.0f;
    }

    input_scale    = input_attr->scale;
    inputZP        = input_attr->zero_point;
    output_scale   = output_attr->scale;
    outputZP       = output_attr->zero_point;

    if (is_run_nx_kernel)
    {
        gpu_param.global_scale[0] = 8;
        gpu_param.global_scale[1] = 1;
    }
    else
    {
        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }

    if (is_run_nx_kernel)
    {
        gpu_dp_inst_t uniResize2xUp_half_2x8 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x32212110, 0x54434332, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3a003400, 0x34003a00, 0x3a003400, 0x34003a00,
            0x3a003400, 0x34003a00, 0x3a003400, 0x34003a00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize2xUp_2x8 = {{
            0x51515151, // TCfg
            0x00000000, // ASelt
            0x21011000, 0x43033202, // ABin
            0xa2a2a2a2, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x38003800, 0x00003c00, 0x38003800,
            0x00003c00, 0x38003800, 0x00003c00, 0x38003800 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize4xUp_half_2x8 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x21211010, 0x32322121, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x39003600, 0x3b003000, 0x30003b00, 0x36003900,
            0x39003600, 0x3b003000, 0x30003b00, 0x36003900 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize4xUp_2x8 = {{
            0x55515551, // TCfg
            0x00000000, // ASelt
            0x10101000, 0x21212101, // ABin
            0xaaa2aaa2, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x34003a00, 0x38003800, 0x3a003400,
            0x00003c00, 0x34003a00, 0x38003800, 0x3a003400 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_half_2x8 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x10101010, 0x21212121, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x38803700, 0x39803500, 0x3a803200, 0x3b802c00,
            0x2c003b80, 0x32003a80, 0x35003980, 0x37003880 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_2x8 = {{
            0x55555551, // TCfg
            0x00000000, // ASelt
            0x10101000, 0x10101010, // ABin
            0xaaaaaaa2, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x30003b00, 0x34003a00, 0x36003900,
            0x38003800, 0x39003600, 0x3a003400, 0x3b003000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize2xDown_8bit_half_2x8 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x38003800, 0x38003800, 0x38003800, 0x38003800,
            0x38003800, 0x38003800, 0x38003800, 0x38003800 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize2xDown_8bit_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x0e0c0a08, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize2xDown_16bit_half_2x8 = {{
            0x55555555, // TCfg
            0x55550000, // ASelt
            0x76543210, 0x76543210, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x38003800, 0x38003800, 0x38003800, 0x38003800,
            0x38003800, 0x38003800, 0x38003800, 0x38003800 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize2xDown_16bit_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        if ( UP_2X_HALF_SAME == scale_flag )
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxUp_2x8", &uniResize2xUp_half_2x8);
        }
        else if ( UP_2X_SAME == scale_flag )
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxUp_2x8", &uniResize2xUp_2x8);
        }
        else if ( UP_4X_HALF_SAME == scale_flag )
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxUp_2x8", &uniResize4xUp_half_2x8);
        }
        else if ( UP_4X_SAME == scale_flag )
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxUp_2x8", &uniResize4xUp_2x8);
        }
        else if ( UP_8X_HALF_SAME == scale_flag )
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxUp_2x8", &uniResize8xUp_half_2x8);
        }
        else if ( UP_8X_SAME == scale_flag )
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxUp_2x8", &uniResize8xUp_2x8);
        }
        else if ( DOWN_2X_HALF_SAME == scale_flag )
        {
            if (I8 == input_dtype || U8 == input_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxDown_2x8", &uniResize2xDown_8bit_half_2x8);
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxDown_2x8", &uniResize2xDown_16bit_half_2x8);
            }
        }
        else if ( DOWN_2X_SAME == scale_flag )
        {
            if (I8 == input_dtype || U8 == input_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxDown_2x8", &uniResize2xDown_8bit_2x8);
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "uniResizeNxDown_2x8", &uniResize2xDown_16bit_2x8);
            }
        }
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        float dfpScale = input_scale / output_scale;
        gpu_dp_inst_t uniConvertDFP2FP32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
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
        vx_uint32 uniConvertDFP2FP32_left_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertDFP2FP32_right_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00030001, 0x00070005, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        if (I8 == input_dtype && I8 == output_dtype && out_width > in_width)
        {
            gpu_dp_inst_t uniConvertI32toI16_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniGetMaskShift_2x8 = {{
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertDFP2FP32_part1_4x4 = {{
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00150004, 0x00370026, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertDFP2FP32_4x4", &uniConvertDFP2FP32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertDFP2FP32_part1_4x4",
                                                 &uniConvertDFP2FP32_part1_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if (I16 == input_dtype && I16 == output_dtype && out_width > in_width)
        {
            gpu_dp_inst_t uniConvertI32toI16_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniGetMaskShift_2x8 = {{
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvertDFP2FP32_part1_4x4 = {{
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00150004, 0x00370026, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertDFP2FP32_4x4", &uniConvertDFP2FP32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertDFP2FP32_part1_4x4",
                                                 &uniConvertDFP2FP32_part1_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertDFP2FP32_left_4x4",
                                                   &uniConvertDFP2FP32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertDFP2FP32_right_4x4",
                                                   &uniConvertDFP2FP32_right_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        status  = vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        status |= vsi_nn_kernel_gpu_add_param( node, "dfpScale", &dfpScale);
        status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
        gpu_param.global_scale[1] = out_height;
    }
    else if (U8 == input_dtype && (U8 == output_dtype || F16 == output_dtype))
    {
        float   uint8Scale             = input_scale / output_scale;
        float   uint8ZP_out            = (float)outputZP;
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
        gpu_dp_inst_t uniU8SubZPtoFp32_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        vx_uint32 uniU8SubZPtoFp32_left_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00020000, 0x00060004, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        };
        vx_uint32 uniU8SubZPtoFp32_right_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00030001, 0x00070005, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        };

        if (F16 == output_dtype)
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uint8Scale", &input_scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_left_4x4",  &uniU8SubZPtoFp32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_right_4x4", &uniU8SubZPtoFp32_right_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else
        {
            if (out_width > in_width)
            {
                gpu_dp_inst_t uniConvertI32toI16_2x8 = {{
                    0x33333333, // TCfg
                    0x11110000, // ASelt
                    0x03020100, 0x03020100, // ABin
                    0x00000000, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00002400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                }, GPU_DP_TYPE_16};
                gpu_dp_inst_t uniGetMaskShift_2x8 = {{
                    0x99999999, // TCfg
                    0x00000000, // ASelt
                    0x03020100, 0x07060504, // ABin
                    0x55555555, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                }, GPU_DP_TYPE_16};
                gpu_dp_inst_t uniU8SubZPtoFp32_part1_4x4 = {{
                    0x09090909, // TCfg
                    0x00000000, // ASelt
                    0x00150004, 0x00370026, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0x00010001, 0x00000000, 0x00010001, 0x00000000,
                    0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
                }, GPU_DP_TYPE_16};
                vx_uint32 uniBilinear_4x4[16] = {
                    0x05050505, // TCfg
                    0x00000000, // ASelt
                    0x00510040, 0x00730062, // ABin
                    0x05050505, // BSelt
                    0x00320010, 0x00760054, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                };
                status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
                if (is_use_scale_kernel)
                {
                    status |= vsi_nn_kernel_gpu_add_param( node, "uniBilinear_4x4", &uniBilinear_4x4);
                }
                else
                {
                    status |= vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_4x4", &uniU8SubZPtoFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param( node,
                              "uniU8SubZPtoFp32_part1_4x4", &uniU8SubZPtoFp32_part1_4x4);
                }
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            else
            {
                status  = vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_left_4x4",
                                                              &uniU8SubZPtoFp32_left_4x4);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_right_4x4",
                                                              &uniU8SubZPtoFp32_right_4x4);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }

            if (!is_use_scale_kernel)
            {
                status  = vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
                status |= vsi_nn_kernel_gpu_add_param( node, "uint8Scale", &uint8Scale);
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &uint8ZP_out);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
        }
        status  = vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        if (!is_use_scale_kernel)
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "input_ZP", &inputZP);
        }
        status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
        gpu_param.global_scale[1] = out_height;

    }
    else if (F16 == input_dtype && (U8 == output_dtype || F16 == output_dtype))
    {
        float   uint8Scale             = 1.0f / output_scale;
        float   uint8ZP_out            = (vx_float32)outputZP;
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
        gpu_dp_inst_t uniFp16toFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniRightSubLeft_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
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
        vx_uint32 uniConvertFp2FP32_left_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertFp2FP32_right_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00030001, 0x00070005, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        if (F16 == input_dtype && F16 == output_dtype && out_width > in_width)
        {
            gpu_dp_inst_t uniConvertI32toI16_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniGetMaskShift_2x8 = {{
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniFp16toFp32_part1_4x4 = {{
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00150004, 0x00370026, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32_4x4", &uniFp16toFp32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32_part1_4x4", &uniFp16toFp32_part1_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniExtactHalf8_2x8", &uniExtactHalf8_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if (F16 == output_dtype)
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniExtactHalf8_2x8", &uniExtactHalf8_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertFp2FP32_left_4x4",
                                                          &uniConvertFp2FP32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertFp2FP32_right_4x4",
                                                          &uniConvertFp2FP32_right_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4", &uniRightSubLeft_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertFp2FP32_left_4x4",
                                                         &uniConvertFp2FP32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertFp2FP32_right_4x4",
                                                         &uniConvertFp2FP32_right_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uint8Scale", &uint8Scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &uint8ZP_out);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        status  = vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
        gpu_param.global_scale[1] = out_height;
    }
    else if (BF16 == input_dtype && BF16 == output_dtype)
    {
        if (out_width > in_width)
        {
            gpu_dp_inst_t uniConvertI32toI16_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniGetMaskShift_2x8 = {{
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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
            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else
        {
            gpu_dp_inst_t uniConvBF16toF32_odd_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x02050004, 0x06070406, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniConvBF16toF32_even_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x03050104, 0x07070506, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_odd_2x8",  &uniConvBF16toF32_odd_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_even_2x8", &uniConvBF16toF32_even_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        status  = vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        status |= vsi_nn_kernel_gpu_add_param( node, "scale_x", &scale_factor);
        CHECK_STATUS_FAIL_GOTO(status, final );
        gpu_param.global_scale[1] = out_height;
    }
    else
    {
        VSILOGE("input or output's format is not support");
        status = VSI_FAILURE;
        goto final;
    }

    if (!is_run_nx_kernel)
    {
        status  = vsi_nn_kernel_gpu_add_param( node, "half_pixel_value", &half_pixel_value);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if (is_run_nx_kernel)
    {
        gpu_param.global_size[0]   = gpu_align_p2((out_width  + \
                                     gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = depth;
        gpu_param.dim              = 2;
    }
    else
    {
        gpu_param.global_size[0]   = gpu_align_p2((out_width  + \
                                     gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = (out_height + gpu_param.global_scale[1] - 1) / gpu_param.global_scale[1];
        gpu_param.global_size[2]   = depth / gpu_param.global_scale[2];
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );
    return status;
} /* _resize_1d_bilinear_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool is_same_type,
    int32_t  align_corners ,
    int32_t  half_pixel_centers,
    vsi_bool *is_run_opt_kernel,
    vsi_bool *is_run_nx_kernel,
    int32_t *scale_flag_value
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _resize_1d_bilinear_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _resize_1d_bilinear_kernel_map );
    vx_param_description_t * param_def  = _resize_1d_bilinear_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _resize_1d_bilinear_kernel_param_def );
    vx_kernel_initialize_f  initializer = _resize_1d_bilinear_initializer;
    uint32_t key = 0;
    uint32_t i   = 0;
    _internal_scale_e scale_flag = UP;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (outputs[0]->attr.size[0] > inputs[0]->attr.size[0])
    {
        scale_flag = UP;

        if (is_same_type)
        {
            scale_flag = UP_OPT;
        }

        if (2 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0])
        {
            if (is_same_type && (!align_corners) && (half_pixel_centers))
            {
                scale_flag = UP_2X_HALF_SAME;
            }
            else if (is_same_type && (!align_corners) && (!half_pixel_centers))
            {
                scale_flag = UP_2X_SAME;
            }
        }
        else if (4 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0])
        {
            if (is_same_type && (!align_corners) && (half_pixel_centers))
            {
                scale_flag = UP_4X_HALF_SAME;
            }
            else if (is_same_type && (!align_corners) && (!half_pixel_centers))
            {
                scale_flag = UP_4X_SAME;
            }
        }
        else if (8 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0])
        {
            if (is_same_type && (!align_corners) && (half_pixel_centers))
            {
                scale_flag = UP_8X_HALF_SAME;
            }
            else if (is_same_type && (!align_corners) && (!half_pixel_centers))
            {
                scale_flag = UP_8X_SAME;
            }
        }
    }
    else
    {
        scale_flag = DOWN;
        if (inputs[0]->attr.size[0] == 2 * outputs[0]->attr.size[0])
        {
            if (is_same_type && (!align_corners) && (half_pixel_centers))
            {
                scale_flag = DOWN_2X_HALF_SAME;
            }
            else if (is_same_type && (!align_corners) && (!half_pixel_centers))
            {
                scale_flag = DOWN_2X_SAME;
            }
        }
    }

    key = RESIZE_1D_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }

    if ((scale_flag > UP_OPT) && (i >= kernel_map_size) && is_same_type)
    {
        scale_flag = UP_OPT;
        key = RESIZE_1D_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
        for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
        {
            if ( kernel_map[i].key == key )
            {
                break;
            }
        }
    }

    if ((UP_OPT == scale_flag) && (i >= kernel_map_size))
    {
        scale_flag = UP;
        key = RESIZE_1D_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
        for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
        {
            if ( kernel_map[i].key == key )
            {
                break;
            }
        }
    }

    if ((scale_flag <= UP) && (scale_flag > DOWN) && (i >= kernel_map_size))
    {
        scale_flag = DOWN;
        key = RESIZE_1D_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
        for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
        {
            if ( kernel_map[i].key == key )
            {
                break;
            }
        }
    }

    if ( i < kernel_map_size )
    {
        if ((scale_flag > UP_OPT) || ((scale_flag > DOWN) && (scale_flag < UP)))
        {
            param_def_size = _RESIZE_1D_NX_KERENL_PARAM_NUM;
            *is_run_nx_kernel = TRUE;
        }
        else if (UP_OPT == scale_flag)
        {
            param_def_size = _RESIZE_1D_BILINEAR_PARAM_NUM;
            *is_run_opt_kernel = TRUE;
        }
        else
        {
            param_def_size = _RESIZE_NO_SCALE_PARAM_NUM;
            *is_run_opt_kernel = FALSE;
        }
        *scale_flag_value = (int32_t)scale_flag;

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


static vsi_nn_tensor_t* _create_scale_tensor
    (
    vsi_nn_graph_t  *graph,
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *output,
    int32_t          align_corners,
    int32_t          half_pixel_centers
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t*  scale         = NULL;
    uint32_t   dims                 = output->attr.dim_num;
    vsi_size_t   batch                = dims > 3 ? output->attr.size[3] : 1;
    vsi_size_t   width                = output->attr.size[0];
    vsi_size_t   sizes[4]             = { 0, 0, 0, 0 };
    vsi_size_t   item_count           = width * 2 * batch;
    vsi_size_t   input_width          = input->attr.size[0];
    vsi_size_t   x                    = 0;
    vsi_size_t   b                    = 0;
    float      width_scale          = 1.0f;
    uint16_t  *scale_data_ptr       = NULL;

    sizes[0] = width * 2;
    sizes[1] = 1;
    sizes[2] = 1;
    sizes[3] = batch;
    if (align_corners && width > 1)
    {
        width_scale = ((vx_float32)(input_width - 1) * 1.0f) / (vx_float32)(width - 1);
    }
    else
    {
        width_scale = ((vx_float32)input_width * 1.0f) / (vx_float32)width;
    }

    scale_data_ptr = (uint16_t *)malloc(item_count * sizeof(uint16_t));
    if (scale_data_ptr == NULL)
    {
        VSILOGE("allocate memory fail at function %s line %d", __FUNCTION__, __LINE__);
        goto OnError;
    }
    memset(scale_data_ptr, 0, item_count * sizeof(vx_uint16));
    for (b = 0; b < batch; b ++)
    {
        for (x = 0; x < width; x ++)
        {
            float     input_w = 0.0f;
            int32_t   w0      = 0;
            size_t  idx     = b * width * 2 + x * 2;
            float     tl      = 0.0f;
            float     tr      = 0.0f;
            if (half_pixel_centers)
            {
                input_w = ((vx_float32)x + 0.5f) * width_scale - 0.5f;
            }
            else
            {
                input_w = x * width_scale;
            }
            w0 = (vx_int32)input_w;
            tl = (1 - (input_w - w0));
            tr = (input_w - w0);

            scale_data_ptr[idx + 0] = fp32_to_fp16(tl);
            scale_data_ptr[idx + 1] = fp32_to_fp16(tr);
        }
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    attr.size[0] = sizes[0];
    attr.size[1] = sizes[1];
    attr.size[2] = sizes[2];
    attr.size[3] = sizes[3];
    attr.dim_num = (batch == 1) ? 2 : 4;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;

    scale = vsi_nn_CreateTensorFromData(graph, (uint8_t *)scale_data_ptr, &attr);
    if (scale_data_ptr)
    {
        free(scale_data_ptr);
        scale_data_ptr = NULL;
    }

OnError:
    return scale;
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
    vsi_nn_kernel_node_param_t node_params[_RESIZE_1D_BILINEAR_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node   = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );
    vsi_bool is_same_type       = vsi_nn_is_same_type(inputs[0], outputs[0]);
    vsi_bool is_run_opt_kernel  = FALSE;
    vsi_bool is_run_nx_kernel   = FALSE;
    vsi_nn_tensor_t*  scale     = NULL;
    int32_t scale_flag_value   = 0;

    status = _query_kernel( kernel, inputs, outputs, is_same_type,
                           align_corners, half_pixel_centers,
                           &is_run_opt_kernel, &is_run_nx_kernel,
                           &scale_flag_value);

    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            size_t node_params_num = _RESIZE_NO_SCALE_PARAM_NUM;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _RESIZE_1D_BILINEAR_PARAM_NUM,
                    inputs, input_num, outputs, output_num );

            if (!is_run_nx_kernel)
            {
                node_params[SCALAR_ALIGN_CORNERS] = vsi_nn_kernel_scalar_create( graph, I32, &align_corners );
                node_params[SCALAR_HALF_PIXEL] = vsi_nn_kernel_scalar_create( graph, I32, &half_pixel_centers );
            }else
            {
               node_params[SCALAR_SCALE_TYPE] = vsi_nn_kernel_scalar_create( graph, I32, &scale_flag_value );
               node_params_num =  _RESIZE_1D_NX_KERENL_PARAM_NUM;
            }


            if (is_run_opt_kernel)
            {
                scale = _create_scale_tensor(graph, inputs[0], outputs[0], align_corners, half_pixel_centers);
                CHECK_PTR_FAIL_GOTO( scale, "Create tensor fail.", final );
                node_params[SCALAR_TENSOR_SCALE] = (vsi_nn_kernel_node_param_t)(scale->t);
                node_params_num = _RESIZE_1D_BILINEAR_PARAM_NUM;
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            if (!is_run_nx_kernel)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_ALIGN_CORNERS] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_HALF_PIXEL] );
            }
            else
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE_TYPE] );
            }
        }
    }

final:
    if (is_run_opt_kernel)
    {
        if (scale)
        {
            vsi_nn_ReleaseTensor(&scale);
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( resize_1d_bilinear, _setup )


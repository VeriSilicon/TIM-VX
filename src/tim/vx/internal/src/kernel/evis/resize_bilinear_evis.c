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
#include "utils/vsi_nn_dtype_util_prv.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    DOWN = 0,
    UP,
    UP_OPT,
    UP_2X_HALF,
    UP_3X_HALF,
    UP_4X_HALF,
    UP_8X_HALF,
} _internal_scale_e;

#define _RESIZE_BILINEAR_KERNEL_SOURCE(_input_type)      "resize_bilinear_"#_input_type
#define _RESIZE_BILINEAR_KERNEL_SOURCE_OPT(_input_type)  "resize_bilinear_"#_input_type"_opt"
#define _RESIZE_BILINEAR_KERNEL_SOURCE_UP_HPC1(_input_type)  "resize_bilinear_"#_input_type"_half_pixel_centers_1"
#define _RESIZE_BILINEAR_KERNEL_SOURCE_UP_HPC2(_input_type)  "resize_bilinear_"#_input_type"_half_pixel_centers_2"

#define STR(a) #a
// Add kernel hashtable here
#define RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, scale_flag ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (scale_flag))

#define PACK_KERNEL_MAP_DOWN( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, DOWN ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_DOWN"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_UP"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_OPT( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_OPT ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_UP_opt"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_OPT(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_2X_HALF( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_2X_HALF ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE) \
            "_SAME_2x_upsample_half_pixel_centers"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_UP_HPC1(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_4X_HALF( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_4X_HALF ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE) \
            "_SAME_4x_upsample_half_pixel_centers"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_UP_HPC1(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_8X_HALF( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_8X_HALF ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE) \
            "_SAME_8x_upsample_half_pixel_centers"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_UP_HPC2(IN_DTYPE) }

#define PACK_KERNEL_MAP_UP_3X_HALF( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_BILINEAR_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_3X_HALF ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_"STR(IN_DTYPE)"to"STR(OUT_DTYPE) \
            "_SAME_3x_upsample_half_pixel_centers"), \
          _RESIZE_BILINEAR_KERNEL_SOURCE_UP_HPC1(IN_DTYPE) }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _resize_bilinear_kernel_map[] =
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
    PACK_KERNEL_MAP_UP_2X_HALF(U8, U8),
    PACK_KERNEL_MAP_UP_3X_HALF(U8, U8),
    PACK_KERNEL_MAP_UP_4X_HALF(U8, U8),
    PACK_KERNEL_MAP_UP_8X_HALF(U8, U8),
};


/*
 * Kernel params
 */
static vx_param_description_t _resize_bilinear_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _RESIZE_BILINEAR_PARAM_NUM  _cnt_of_array( _resize_bilinear_kernel_param_def )
#define _RESIZE_NO_SCALE_PARAM_NUM  4

#define SCALAR_ALIGN_CORNERS         (2)
#define SCALAR_HALF_PIXEL            (3)
#define SCALAR_TENSOR_SCALE          (4)

static vsi_bool _is_same_quant
    (
    vsi_nn_kernel_tensor_attr_t * input0_attr,
    vsi_nn_kernel_tensor_attr_t * input1_attr
    )
{
    if (NULL == input0_attr || NULL == input1_attr)
    {
        return FALSE;
    }

    if (input0_attr->dtype != input1_attr->dtype || input0_attr->quant != input1_attr->quant)
    {
        return FALSE;
    }
    if (input0_attr->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        if (input0_attr->dfp.fl != input1_attr->dfp.fl)
        {
            return FALSE;
        }
    }
    else if (input0_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        const float diff = (float)1e-5;
        if (input0_attr->asymm.zero_point != input1_attr->asymm.zero_point)
        {
            return FALSE;
        }
        if(vsi_nn_float_compare(input0_attr->asymm.scale, input1_attr->asymm.scale, diff) == FALSE)
        {
            return FALSE;
        }
    }

    return TRUE;
} /* _is_same_quant */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_resize_bilinear_initializer)
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
    int32_t align_corners;
    int32_t half_pixel_centers;

    uint32_t    depth              = 0;
    int32_t     srcFixPointPos     = 0;
    int32_t     dstFixPointPos     = 0;
    float       input_scale        = 1.0;
    int32_t     inputZP            = 0;
    float       output_scale       = 1.0;
    int32_t     outputZP           = 0;
    float       scale_factor[2];
    uint32_t    in_width;
    uint32_t    in_height;
    uint32_t    out_width;
    uint32_t    out_height;
    float       half_pixel_value = 0.0f;
    vsi_bool    is_use_scale_kernel = (vsi_bool)(_RESIZE_BILINEAR_PARAM_NUM == param_size);
    vsi_bool    is_half_pixel_centers     = FALSE;
    vsi_bool    is_2x_up_kernel  = FALSE;
    vsi_bool    is_3x_up_kernel  = FALSE;
    vsi_bool    is_4x_up_kernel  = FALSE;
    vsi_bool    is_8x_up_kernel  = FALSE;

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
    in_height         = (uint32_t)(in_shape->data[1]);
    depth             = (uint32_t)(in_shape->data[2]);
    out_width         = (uint32_t)(out_shape->data[0]);
    out_height        = (uint32_t)(out_shape->data[1]);

    if (align_corners && out_width > 1)
    {
        scale_factor[0] = ((vx_float32)(in_width - 1) * 1.0f) / (vx_float32)(out_width - 1);
    }
    else
    {
        scale_factor[0] = ((vx_float32)in_width * 1.0f) / (vx_float32)out_width;
    }

    if (align_corners && out_height > 1)
    {
        scale_factor[1] = ((vx_float32)(in_height - 1) * 1.0f) / (vx_float32)(out_height - 1);
    }
    else
    {
        scale_factor[1] = ((vx_float32)in_height * 1.0f) / (vx_float32)out_height;
    }

    if (half_pixel_centers)
    {
        half_pixel_value = 0.5f;
    }
    else
    {
        half_pixel_value = 0.0f;
    }

    is_half_pixel_centers = (!align_corners) && (half_pixel_centers);

    if ((U8 == input_dtype) && (_is_same_quant(input_attr, output_attr)) && is_half_pixel_centers)
    {
        is_2x_up_kernel = (2 * in_width == out_width) && (2 * in_height == out_height);
        is_3x_up_kernel = (3 * in_width == out_width) && (3 * in_height == out_height);
        is_4x_up_kernel = (4 * in_width == out_width) && (4 * in_height == out_height);
        is_8x_up_kernel = (8 * in_width == out_width) && (8 * in_height == out_height);
    }

    if (U8 == input_dtype && VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant )
    {
        input_scale    = input_attr->asymm.scale;
        inputZP        = input_attr->asymm.zero_point;
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        srcFixPointPos   = input_attr->dfp.fl;
        if (srcFixPointPos >= 0)
        {
            input_scale = 1.0f / (vx_float32) ((int64_t)1 << srcFixPointPos);
        }
        else if (srcFixPointPos < 0)
        {
            input_scale = (vx_float32)((int64_t)1 << -srcFixPointPos);
        }
        inputZP = 0;
    }
    else
    {
        input_scale = 1.0f;
        inputZP     = 0;
    }

    if (U8 == output_dtype && VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant )
    {
        output_scale   = output_attr->asymm.scale;
        outputZP       = output_attr->asymm.zero_point;
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        dstFixPointPos = output_attr->dfp.fl;
        if (dstFixPointPos >= 0)
        {
            output_scale = (vx_float32) ((int64_t)1 << dstFixPointPos);
        }
        else if (dstFixPointPos < 0)
        {
            output_scale = 1.0f / (vx_float32) ((int64_t)1 << -dstFixPointPos);
        }
        outputZP = 0;
    }
    else
    {
        output_scale = 1.0;
        outputZP     = 0;
    }

    if (is_2x_up_kernel || is_4x_up_kernel || is_8x_up_kernel)
    {
        gpu_param.global_scale[0] = 16;
        gpu_param.global_scale[1] = 1;
    }
    else if (is_3x_up_kernel)
    {
        gpu_param.global_scale[0] = 15;
        gpu_param.global_scale[1] = 6;
        gpu_param.global_scale[2] = 1;
    }
    else
    {
        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }

    if (is_2x_up_kernel)
    {
        gpu_dp_inst_t uniResize2xUp_0_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x4418c020, 0x29444194, 0xc8629c86, 0x83a4c839, 0xad0a4a4c, // BinSelect
            0x00000704, // AccumType, ConstantType, and PostShift
            0x09030301, 0x03090103, 0x09030301, 0x03090103,
            0x09030301, 0x03090103, 0x09030301, 0x03090103 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize2xUp_1_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x4c5ad0a4, 0x6b54c5b5, 0xd8e6bd8e, 0x07c5d07b, 0xce128c5d, // BinSelect
            0x00000704, // AccumType, ConstantType, and PostShift
            0x09030301, 0x03090103, 0x09030301, 0x03090103,
            0x09030301, 0x03090103, 0x09030301, 0x03090103 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize2xUp_0_4x8", &uniResize2xUp_0_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize2xUp_1_4x8", &uniResize2xUp_1_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_3x_up_kernel)
    {
        gpu_dp_inst_t uniResize3xUp_l00_2x8 = {{
            0x15515515, // TCfg
            0x00000000, // ASelt
            0x21210110, 0x03323202, // ABin
            0x2aa2aa2a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000610, // AccumType, ConstantType, and PostShift
            0xaaaa5555, 0x0000ffff, 0x5555aaaa, 0xaaaa5555,
            0x0000ffff, 0x5555aaaa, 0xaaaa5555, 0x0000ffff // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize3xUp_l01_2x8 = {{
            0x05155155, // TCfg
            0x00000000, // ASelt
            0x54044343, 0x00650554, // ABin
            0x0a2aa2aa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000610, // AccumType, ConstantType, and PostShift
            0x5555aaaa, 0xaaaa5555, 0x0000ffff, 0x5555aaaa,
            0xaaaa5555, 0x0000ffff, 0x5555aaaa, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize3xUp_l10_4x4 = {{
            0x55551155, // TCfg
            0x50501050, // ASelt
            0x01011010, 0x21212121, // ABin
            0xaaaa22aa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x38e41c72, 0x1c720e39, 0x00005556, 0x00002aab,
            0x1c7238e4, 0x0e391c72, 0x38e41c72, 0x1c720e39 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize3xUp_l11_4x4 = {{
            0x11555511, // TCfg
            0x10505010, // ASelt
            0x32320202, 0x03033232, // ABin
            0x22aaaa22, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x00005556, 0x00002aab, 0x1c7238e4, 0x0e391c72,
            0x38e41c72, 0x1c720e39, 0x00005556, 0x00002aab // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize3xUp_l12_4x4 = {{
            0x55115555, // TCfg
            0x50105050, // ASelt
            0x43434343, 0x54540404, // ABin
            0xaa22aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x1c7238e4, 0x0e391c72, 0x38e41c72, 0x1c720e39,
            0x00005556, 0x00002aab, 0x1c7238e4, 0x0e391c72 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize3xUp_l13_4x4 = {{
            0x00551155, // TCfg
            0x00501050, // ASelt
            0x05055454, 0x00006565, // ABin
            0x00aa22aa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x38e41c72, 0x1c720e39, 0x00005556, 0x00002aab,
            0x1c7238e4, 0x0e391c72, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize3xUp_l00_2x8", &uniResize3xUp_l00_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize3xUp_l01_2x8", &uniResize3xUp_l01_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize3xUp_l10_4x4", &uniResize3xUp_l10_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize3xUp_l11_4x4", &uniResize3xUp_l11_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize3xUp_l12_4x4", &uniResize3xUp_l12_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize3xUp_l13_4x4", &uniResize3xUp_l13_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_4x_up_kernel)
    {
        gpu_dp_inst_t uniResize4xUp_l00_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x0208c020, 0x1944418c, 0x44419444, 0x62944419, 0x9c8629c8, // BinSelect
            0x00000406, // AccumType, ConstantType, and PostShift
            0x190f0f09, 0x23051503, 0x05230315, 0x0f19090f,
            0x190f0f09, 0x23051503, 0x05230315, 0x0f19090f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize4xUp_l01_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x8629c862, 0x3a4c839c, 0x4c83a4c8, 0xa4a4c83a, 0xad0a4ad0, // BinSelect
            0x00000406, // AccumType, ConstantType, and PostShift
            0x190f0f09, 0x23051503, 0x05230315, 0x0f19090f,
            0x190f0f09, 0x23051503, 0x05230315, 0x0f19090f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize4xUp_l10_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x0208c020, 0x1944418c, 0x44419444, 0x62944419, 0x9c8629c8, // BinSelect
            0x00000406, // AccumType, ConstantType, and PostShift
            0x23150503, 0x31070701, 0x07310107, 0x15230305,
            0x23150503, 0x31070701, 0x07310107, 0x15230305 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize4xUp_l11_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x8629c862, 0x3a4c839c, 0x4c83a4c8, 0xa4a4c83a, 0xad0a4ad0, // BinSelect
            0x00000406, // AccumType, ConstantType, and PostShift
            0x23150503, 0x31070701, 0x07310107, 0x15230305,
            0x23150503, 0x31070701, 0x07310107, 0x15230305 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize4xUp_l00_4x8", &uniResize4xUp_l00_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize4xUp_l01_4x8", &uniResize4xUp_l01_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize4xUp_l10_4x8", &uniResize4xUp_l10_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize4xUp_l11_4x8", &uniResize4xUp_l11_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_8x_up_kernel)
    {
        gpu_dp_inst_t uniResize8xUp_l00_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x0208c020, 0x08c0208c, 0x44418c02, 0x41944419, 0x94441944, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x513f3f31, 0x632d4d23, 0x751b5b15, 0x87096907,
            0x09870769, 0x1b75155b, 0x2d63234d, 0x3f51313f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l01_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x44194441, 0x19444194, 0xc8629444, 0x629c8629, 0x9c8629c8, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x513f3f31, 0x632d4d23, 0x751b5b15, 0x87096907,
            0x09870769, 0x1b75155b, 0x2d63234d, 0x3f51313f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l10_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x0208c020, 0x08c0208c, 0x44418c02, 0x41944419, 0x94441944, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x634d2d23, 0x79373719, 0x8f21410f, 0xa50b4b05,
            0x0ba5054b, 0x218f0f41, 0x37791937, 0x4d63232d // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l11_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x44194441, 0x19444194, 0xc8629444, 0x629c8629, 0x9c8629c8, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x634d2d23, 0x79373719, 0x8f21410f, 0xa50b4b05,
            0x0ba5054b, 0x218f0f41, 0x37791937, 0x4d63232d // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l20_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x0208c020, 0x08c0208c, 0x44418c02, 0x41944419, 0x94441944, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x755b1b15, 0x8f41210f, 0xa9272709, 0xc30d2d03,
            0x0dc3032d, 0x27a90927, 0x418f0f21, 0x5b75151b // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l21_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x44194441, 0x19444194, 0xc8629444, 0x629c8629, 0x9c8629c8, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x755b1b15, 0x8f41210f, 0xa9272709, 0xc30d2d03,
            0x0dc3032d, 0x27a90927, 0x418f0f21, 0x5b75151b // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l30_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x0208c020, 0x08c0208c, 0x44418c02, 0x41944419, 0x94441944, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x87690907, 0xa54b0b05, 0xc32d0d03, 0xe10f0f01,
            0x0fe1010f, 0x2dc3030d, 0x4ba5050b, 0x69870709 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize8xUp_l31_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x44194441, 0x19444194, 0xc8629444, 0x629c8629, 0x9c8629c8, // BinSelect
            0x00000708, // AccumType, ConstantType, and PostShift
            0x87690907, 0xa54b0b05, 0xc32d0d03, 0xe10f0f01,
            0x0fe1010f, 0x2dc3030d, 0x4ba5050b, 0x69870709 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l00_4x8", &uniResize8xUp_l00_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l01_4x8", &uniResize8xUp_l01_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l10_4x8", &uniResize8xUp_l10_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l11_4x8", &uniResize8xUp_l11_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l20_4x8", &uniResize8xUp_l20_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l21_4x8", &uniResize8xUp_l21_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l30_4x8", &uniResize8xUp_l30_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize8xUp_l31_4x8", &uniResize8xUp_l31_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        float dfpScale = input_scale * output_scale;
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
        gpu_dp_inst_t uniRightSubLeft_4x4 = {{
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDFPtoFp32_left_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

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
            status |= vsi_nn_kernel_gpu_add_param( node, "uniDFPtoFp32_left_4x4", &uniConvertDFP2FP32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4",
                                                 &uniConvertDFP2FP32_part1_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "depth", &depth);
            CHECK_STATUS_FAIL_GOTO(status, final );

            gpu_param.global_scale[2] = depth;
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
            status |= vsi_nn_kernel_gpu_add_param( node, "uniDFPtoFp32_left_4x4", &uniConvertDFP2FP32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4",
                                                 &uniConvertDFP2FP32_part1_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "depth", &depth);
            CHECK_STATUS_FAIL_GOTO(status, final );

            gpu_param.global_scale[2] = depth;
        }
        else
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4", &uniRightSubLeft_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniDFPtoFp32_left_4x4", &uniDFPtoFp32_left_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        status  = vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "scale_xy", scale_factor);
        status |= vsi_nn_kernel_gpu_add_param( node, "dfpScale", &dfpScale);
        CHECK_STATUS_FAIL_GOTO(status, final );
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
        gpu_dp_inst_t uniU8SubZPtoFp32_left_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00020000, 0x00060004, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8RightSubLeft_4x4 = {{
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};


        if (F16 == output_dtype)
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uint8Scale", &input_scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_left_4x4", &uniU8SubZPtoFp32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniU8RightSubLeft_4x4", &uniU8RightSubLeft_4x4);
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
                gpu_dp_inst_t uniBilinear_4x4_b = {{
                    0x55555555, // TCfg
                    0x55550000, // ASelt
                    0x76543210, 0x76543210, // ABin
                    0x00000000, // BSelt
                    0xd951c840, 0xfb73ea62, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                }, GPU_DP_TYPE_16};
                status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
                status |= vsi_nn_kernel_gpu_add_param( node, "depth", &depth);
                if (is_use_scale_kernel)
                {
                    status |= vsi_nn_kernel_gpu_add_param( node, "uniBilinear_4x4_b", &uniBilinear_4x4_b);
                }
                else
                {
                    status  = vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_left_4x4", &uniU8SubZPtoFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param( node,
                              "uniU8RightSubLeft_4x4", &uniU8SubZPtoFp32_part1_4x4);
                }
                CHECK_STATUS_FAIL_GOTO(status, final );

                gpu_param.global_scale[2] = depth;
            }
            else if (!is_use_scale_kernel)
            {
                status = vsi_nn_kernel_gpu_add_param( node, "uniU8SubZPtoFp32_left_4x4", &uniU8SubZPtoFp32_left_4x4);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniU8RightSubLeft_4x4", &uniU8RightSubLeft_4x4);
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
        status  = vsi_nn_kernel_gpu_add_param( node, "scale_xy", scale_factor);
        if (!is_use_scale_kernel)
        {
            status = vsi_nn_kernel_gpu_add_param( node, "input_ZP", &inputZP);
        }
        CHECK_STATUS_FAIL_GOTO(status, final );
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
        gpu_dp_inst_t uniRightSubLeft_4x4 = {{
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniFp16toFp32_left_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
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
            gpu_dp_inst_t uniFp16toFp32_Lo_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniFp16toFp32_Hi_4x4 = {{
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
            status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32_left_4x4", &uniFp16toFp32_Lo_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4", &uniFp16toFp32_Hi_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniExtactHalf8_2x8", &uniExtactHalf8_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "depth", &depth);
            CHECK_STATUS_FAIL_GOTO(status, final );

            gpu_param.global_scale[2] = depth;
        }
        else if (F16 == output_dtype)
        {
            status = vsi_nn_kernel_gpu_add_param( node, "uniExtactHalf8_2x8", &uniExtactHalf8_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4", &uniRightSubLeft_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32_left_4x4", &uniFp16toFp32_left_4x4);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else
        {
            status  = vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bit_2x8", &uniExtact8Bit_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniRightSubLeft_4x4", &uniRightSubLeft_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32_left_4x4", &uniFp16toFp32_left_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "uint8Scale", &uint8Scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &uint8ZP_out);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        status = vsi_nn_kernel_gpu_add_param( node, "scale_xy", scale_factor);
        CHECK_STATUS_FAIL_GOTO(status, final );
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
            status  = vsi_nn_kernel_gpu_add_param( node, "scale_xy", scale_factor);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertI32toI16_2x8", &uniConvertI32toI16_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniGetMaskShift_2x8", &uniGetMaskShift_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "depth", &depth);
            CHECK_STATUS_FAIL_GOTO(status, final );

            gpu_param.global_scale[2] = depth;
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

            status  = vsi_nn_kernel_gpu_add_param( node, "scale_xy", scale_factor);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_odd_2x8",  &uniConvBF16toF32_odd_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvBF16toF32_even_2x8", &uniConvBF16toF32_even_2x8);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
    }
    else
    {
        VSILOGE("input or output's format is not support");
        status = VSI_FAILURE;
        goto final;
    }

    if (!is_2x_up_kernel && !is_3x_up_kernel && !is_4x_up_kernel&& !is_8x_up_kernel)
    {
        status  = vsi_nn_kernel_gpu_add_param( node, "half_pixel_value", &half_pixel_value);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if (is_2x_up_kernel || is_4x_up_kernel || is_8x_up_kernel)
    {
        gpu_param.global_size[0] = gpu_align_p2((out_width  + \
                                   gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1] = depth;
        gpu_param.dim            = 2;
    }
    else
    {
        gpu_param.global_size[0] = gpu_align_p2((out_width  + \
                                   gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1] = (out_height + gpu_param.global_scale[1] - 1) / gpu_param.global_scale[1];
        gpu_param.global_size[2] = depth / gpu_param.global_scale[2];
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );
    return status;
} /* _resize_bilinear_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool is_same_type,
    vsi_bool is_evis2,
    int32_t align_corners,
    int32_t half_pixel_centers,
    vsi_bool *is_run_opt_kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _resize_bilinear_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _resize_bilinear_kernel_map );
    vx_param_description_t * param_def  = _resize_bilinear_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _resize_bilinear_kernel_param_def );
    vx_kernel_initialize_f  initializer = _resize_bilinear_initializer;
    uint32_t key;
    uint32_t i;
    vsi_bool is_2x_upsample =(2 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0]) \
                    && (2 * inputs[0]->attr.size[1] == outputs[0]->attr.size[1]);
    vsi_bool is_3x_upsample =(3 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0]) \
                    && (3 * inputs[0]->attr.size[1] == outputs[0]->attr.size[1]);
    vsi_bool is_4x_upsample =(4 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0]) \
                    && (4 * inputs[0]->attr.size[1] == outputs[0]->attr.size[1]);
    vsi_bool is_8x_upsample =(8 * inputs[0]->attr.size[0] == outputs[0]->attr.size[0]) \
                    && (8 * inputs[0]->attr.size[1] == outputs[0]->attr.size[1]);
    _internal_scale_e scale_flag = UP;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    is_2x_upsample &= (in_dtype == U8);
    is_3x_upsample &= (in_dtype == U8);
    is_4x_upsample &= (in_dtype == U8);
    is_8x_upsample &= (in_dtype == U8);

    if (outputs[0]->attr.size[0] > inputs[0]->attr.size[0])
    {
        if (is_same_type && (!align_corners) && (half_pixel_centers) && is_2x_upsample)
        {
            scale_flag = UP_2X_HALF;
        }
        else if (is_same_type && (!align_corners) && (half_pixel_centers) && is_3x_upsample)
        {
            scale_flag = UP_3X_HALF;
        }
        else if (is_same_type && (!align_corners) && (half_pixel_centers) && is_4x_upsample)
        {
            scale_flag = UP_4X_HALF;
        }
        else if (is_same_type && (!align_corners) && (half_pixel_centers) && is_8x_upsample)
        {
            scale_flag = UP_8X_HALF;
        }
        else if (is_same_type && is_evis2)
        {
            scale_flag = UP_OPT;
        }
        else
        {
            scale_flag = UP;
        }
    }
    else
    {
        scale_flag = DOWN;
    }

    key = RESIZE_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
    for( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if ((UP_OPT == scale_flag) && (i >= kernel_map_size))
    {
        scale_flag = UP;
        key = RESIZE_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
        for( i = 0; i < (uint32_t)kernel_map_size; i ++ )
        {
            if( kernel_map[i].key == key )
            {
                break;
            }
        }
    }

    if ((UP == scale_flag) && (i >= kernel_map_size))
    {
        scale_flag = DOWN;
        key = RESIZE_BILINEAR_HASH_KEY( in_dtype, out_dtype, scale_flag );
        for( i = 0; i < (uint32_t)kernel_map_size; i ++ )
        {
            if( kernel_map[i].key == key )
            {
                break;
            }
        }
    }

    if( i < kernel_map_size )
    {
        if (UP_OPT == scale_flag)
        {
            param_def_size = _RESIZE_BILINEAR_PARAM_NUM;
            *is_run_opt_kernel = TRUE;
        }
        else
        {
            param_def_size = _RESIZE_NO_SCALE_PARAM_NUM;
            *is_run_opt_kernel = FALSE;
        }
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
    vsi_size_t   width                = output->attr.size[0];
    vsi_size_t   height               = output->attr.size[1];
    vsi_size_t   batch                = dims > 3 ? output->attr.size[3] : 1;
    vsi_size_t   sizes[4]             = {width * 4, height, 1, batch};
    vsi_size_t   item_count           = width * 4 * height * batch;
    vsi_size_t   input_width          = input->attr.size[0];
    vsi_size_t   input_height         = input->attr.size[1];
    vsi_size_t   x                    = 0;
    vsi_size_t   y                    = 0;
    vsi_size_t   b                    = 0;
    float      width_scale          = 1.0f;
    float      height_scale         = 1.0f;
    uint16_t  *scale_data_ptr       = NULL;

    if (align_corners && width > 1)
    {
        width_scale = ((vx_float32)(input_width - 1) * 1.0f) / (vx_float32)(width - 1);
    }
    else
    {
        width_scale = ((vx_float32)input_width * 1.0f) / (vx_float32)width;
    }

    if (align_corners && height > 1)
    {
        height_scale = ((vx_float32)(input_height - 1) * 1.0f) / (vx_float32)(height - 1);
    }
    else
    {
        height_scale = ((vx_float32)input_height * 1.0f) / (vx_float32)height;
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
        for (y = 0; y < height; y ++)
        {
            float    input_h = 0.0f;
            int32_t  h0      = 0;
            if (half_pixel_centers)
            {
                input_h = ((vx_float32)y + 0.5f) * height_scale - 0.5f;
            }
            else
            {
                input_h = y * height_scale;
            }
            h0 = (int32_t)input_h;
            for (x = 0; x < width; x ++)
            {
                float     input_w = 0.0f;
                int32_t   w0      = 0;
                vsi_size_t  idx     = b * width * 4 * height + y * width * 4 + x * 4;
                float     tl      = 0.0f;
                float     tr      = 0.0f;
                float     bl      = 0.0f;
                float     br      = 0.0f;
                if (half_pixel_centers)
                {
                    input_w = ((vx_float32)x + 0.5f) * width_scale - 0.5f;
                }
                else
                {
                    input_w = x * width_scale;
                }
                w0 = (vx_int32)input_w;
                tl = (1 - (input_h - h0)) * (1 - (input_w - w0));
                tr = (1 - (input_h - h0)) * (input_w - w0);
                bl = (input_h - h0) * (1 - (input_w - w0));
                br = (input_h - h0) * (input_w - w0);

                scale_data_ptr[idx + 0] = fp32_to_fp16(tl);
                scale_data_ptr[idx + 1] = fp32_to_fp16(tr);
                scale_data_ptr[idx + 2] = fp32_to_fp16(bl);
                scale_data_ptr[idx + 3] = fp32_to_fp16(br);
            }
        }
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    attr.size[0] = sizes[0];
    attr.size[1] = sizes[1];
    attr.size[2] = sizes[2];
    attr.size[3] = sizes[3];
    attr.dim_num = dims;
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
    vsi_nn_kernel_node_param_t node_params[_RESIZE_BILINEAR_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node   = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );
    vsi_bool is_same_type       = vsi_nn_is_same_type(inputs[0], outputs[0]);
    vsi_bool is_evis2           = (vsi_bool)(graph->ctx->config.evis.ver == VSI_NN_HW_EVIS_2);
    vsi_bool is_run_opt_kernel  = FALSE;
    vsi_nn_tensor_t*  scale     = NULL;

    status = _query_kernel( kernel, inputs, outputs, is_same_type, is_evis2,
                            align_corners, half_pixel_centers, &is_run_opt_kernel);
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            size_t node_params_num = _RESIZE_NO_SCALE_PARAM_NUM;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _RESIZE_BILINEAR_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_ALIGN_CORNERS] = vsi_nn_kernel_scalar_create( graph, I32, &align_corners );
            node_params[SCALAR_HALF_PIXEL] = vsi_nn_kernel_scalar_create( graph, I32, &half_pixel_centers );
            if (is_run_opt_kernel)
            {
                scale = _create_scale_tensor(graph, inputs[0], outputs[0], align_corners, half_pixel_centers);
                node_params[SCALAR_TENSOR_SCALE] = (vsi_nn_kernel_node_param_t)(scale->t);
                node_params_num = _RESIZE_BILINEAR_PARAM_NUM;
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_ALIGN_CORNERS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_HALF_PIXEL] );
            if (is_run_opt_kernel)
            {
                if (scale)
                {
                    vsi_nn_ReleaseTensor(&scale);
                }
            }
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( resize_bilinear, _setup )

/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_dtype_util_prv.h"

__BEGIN_DECLS

#define STR(a) #a
// Add kernel hashtable here
#define RESIZE_BILINEAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, H_PIXEL_CENTERS, ALIGN_CORNERS, UP_SCALE ) \
        (( IN_DTYPE ) | ( OUT_DTYPE << 8) | (H_PIXEL_CENTERS << 16) | (ALIGN_CORNERS << 17) | (UP_SCALE << 18))

#define BILINEAR_NHWC_PACK_KERNEL_MAP_UP_SCALE( IN_DTYPE, OUT_DTYPE, H_PIXEL_CENTERS, ALIGN_CORNERS, UP_SCALE ) \
        { RESIZE_BILINEAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, H_PIXEL_CENTERS, ALIGN_CORNERS, UP_SCALE ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_nhwc_"STR(IN_DTYPE)"to"STR(OUT_DTYPE) \
            "_"STR(UP_SCALE)"x_upsample_half_pixel_centers"), \
          "resize_bilinear_nhwc" }

#define BILINEAR_NHWC_BOUND_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_SCALE ) \
        (( IN_DTYPE ) | ( OUT_DTYPE << 8) | (UP_SCALE << 16))

#define BILINEAR_NHWC_BOUND_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, UP_SCALE ) \
        { BILINEAR_NHWC_BOUND_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_SCALE ), \
          CVIVANTE_NAMESPACE("evis.resize_bilinear_nhwc_bound_"STR(IN_DTYPE)"to"STR(OUT_DTYPE) \
            "_"STR(UP_SCALE)"x"), \
          "resize_bilinear_nhwc_bound" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _resize_bilinear_nhwc_kernel_map[] =
{
    BILINEAR_NHWC_PACK_KERNEL_MAP_UP_SCALE(U8, U8, 1, 0, 2),
    BILINEAR_NHWC_PACK_KERNEL_MAP_UP_SCALE(U8, U8, 1, 0, 3),
    BILINEAR_NHWC_PACK_KERNEL_MAP_UP_SCALE(U8, U8, 1, 0, 4),
};

static const _kernel_map_type _bilinear_nhwc_bound_kernel_map[] =
{
    BILINEAR_NHWC_BOUND_KERNEL_MAP(U8, U8, 2),
    BILINEAR_NHWC_BOUND_KERNEL_MAP(U8, U8, 3),
    BILINEAR_NHWC_BOUND_KERNEL_MAP(U8, U8, 4),
};

/*
 * Kernel params
 */
static vx_param_description_t _resize_bilinear_nhwc_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _RESIZE_BILINEAR_NHWC_PARAM_NUM  _cnt_of_array( _resize_bilinear_nhwc_kernel_param_def )

#define SCALAR_ALIGN_CORNERS         (2)
#define SCALAR_HALF_PIXEL            (3)

static vx_param_description_t _bilinear_nhwc_bound_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _BILINEAR_NHWC_BOUND_PARAM_NUM  _cnt_of_array( _bilinear_nhwc_bound_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_resize_bilinear_nhwc_initializer)
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
    int32_t align_corners = 0;
    int32_t half_pixel_centers = 0;
    uint32_t    in_width;
    uint32_t    in_height;
    uint32_t    out_width;
    uint32_t    out_height;
    vsi_bool    is_half_pixel_centers     = FALSE;
    vsi_bool    is_2x_up_kernel  = FALSE;
    vsi_bool    is_3x_up_kernel  = FALSE;
    vsi_bool    is_4x_up_kernel  = FALSE;

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

    in_width          = (uint32_t)(in_shape->data[0]);
    in_height         = (uint32_t)(in_shape->data[1]);
    out_width         = (uint32_t)(out_shape->data[0]);
    out_height        = (uint32_t)(out_shape->data[1]);

    is_half_pixel_centers = (!align_corners) && (half_pixel_centers);

    if (is_half_pixel_centers)
    {
        is_2x_up_kernel = (2 * in_width == out_width) && (2 * in_height == out_height);
        is_3x_up_kernel = (3 * in_width == out_width) && (3 * in_height == out_height);
        is_4x_up_kernel = (4 * in_width == out_width) && (4 * in_height == out_height);
    }

    if (is_2x_up_kernel)
    {
        gpu_param.global_scale[0] = 16;
        gpu_param.global_scale[1] = 4;
    }
    else if (is_4x_up_kernel)
    {
        gpu_param.global_scale[0] = 16;
        gpu_param.global_scale[1] = 8;
    }
    else if (is_3x_up_kernel)
    {
        gpu_param.global_scale[0] = 30;
        gpu_param.global_scale[1] = 6;
    }
    else
    {
        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
    }

    if (is_2x_up_kernel)
    {
        gpu_dp_inst_t uniResize_x2_nhwc2_0_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x46194040, 0x3a48829c, 0x4882acca, 0xc4acca3a, 0xbd4e5b50, // BinSelect
            0x00000704, // AccumType, ConstantType, and PostShift
            0x09030301, 0x09030301, 0x03090103, 0x03090103,
            0x09030301, 0x09030301, 0x03090103, 0x03090103 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x2_nhwc2_1_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x4e5b50c4, 0x7c5906bd, 0x5906cdd2, 0x48cdd27c, 0xde569d61, // BinSelect
            0x00000704, // AccumType, ConstantType, and PostShift
            0x09030301, 0x09030301, 0x03090103, 0x03090103,
            0x09030301, 0x09030301, 0x03090103, 0x03090103 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize_x2_nhwc2_0_4x8", &uniResize_x2_nhwc2_0_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x2_nhwc2_1_4x8", &uniResize_x2_nhwc2_1_4x8);
        //status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_3x_up_kernel)
    {
        gpu_dp_inst_t uniResize_x3_nhwc2_l10_4x4 = {{
            0x05055555, // TCfg
            0x04045050, // ASelt
            0x31312020, 0x00330022, // ABin
            0x0a0aaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x38e41c72, 0x1c720e39, 0x38e41c72, 0x1c720e39,
            0x2aab5556, 0x00000000, 0x2aab5556, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l11_4x4 = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x53534242, 0x53534242, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x1c7238e4, 0x0e391c72, 0x1c7238e4, 0x0e391c72,
            0x38e41c72, 0x1c720e39, 0x38e41c72, 0x1c720e39 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l12_4x4 = {{
            0x55550505, // TCfg
            0x50500404, // ASelt
            0x00550044, 0x75756464, // ABin
            0xaaaa0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x2aab5556, 0x00000000, 0x2aab5556, 0x00000000,
            0x1c7238e4, 0x0e391c72, 0x1c7238e4, 0x0e391c72 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l13_4x4 = {{
            0x05055555, // TCfg
            0x04045050, // ASelt
            0x75756464, 0x00770066, // ABin
            0x0a0aaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x38e41c72, 0x1c720e39, 0x38e41c72, 0x1c720e39,
            0x2aab5556, 0x00000000, 0x2aab5556, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l14_4x4 = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x97978686, 0x97978686, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x1c7238e4, 0x0e391c72, 0x1c7238e4, 0x0e391c72,
            0x38e41c72, 0x1c720e39, 0x38e41c72, 0x1c720e39 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l15_4x4 = {{
            0x55550505, // TCfg
            0x50500404, // ASelt
            0x00990088, 0xb9b9a8a8, // ABin
            0xaaaa0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x2aab5556, 0x00000000, 0x2aab5556, 0x00000000,
            0x1c7238e4, 0x0e391c72, 0x1c7238e4, 0x0e391c72 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l16_4x4 = {{
            0x05055555, // TCfg
            0x04045050, // ASelt
            0xb9b9a8a8, 0x00bb00aa, // ABin
            0x0a0aaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x38e41c72, 0x1c720e39, 0x38e41c72, 0x1c720e39,
            0x2aab5556, 0x00000000, 0x2aab5556, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l17_4x4 = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0xdbdbcaca, 0xdbdbcaca, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x1c7238e4, 0x0e391c72, 0x1c7238e4, 0x0e391c72,
            0x38e41c72, 0x1c720e39, 0x38e41c72, 0x1c720e39 // Constant
        }, GPU_DP_TYPE_16};


        gpu_dp_inst_t uniResize_x3_nhwc2_l00_2x8 = {{
            0x55551155, // TCfg
            0x00000000, // ASelt
            0x03023120, 0x53425342, // ABin
            0xaaaa22aa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000610, // AccumType, ConstantType, and PostShift
            0xaaaa5555, 0xaaaa5555, 0x0000ffff, 0x0000ffff,
            0x5555aaaa, 0x5555aaaa, 0xaaaa5555, 0xaaaa5555 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l01_2x8 = {{
            0x11555511, // TCfg
            0x00000000, // ASelt
            0x75640504, 0x07067564, // ABin
            0x22aaaa22, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000610, // AccumType, ConstantType, and PostShift
            0x0000ffff, 0x0000ffff, 0x5555aaaa, 0x5555aaaa,
            0xaaaa5555, 0xaaaa5555, 0x0000ffff, 0x0000ffff // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l02_2x8 = {{
            0x55115555, // TCfg
            0x00000000, // ASelt
            0x97869786, 0xb9a80908, // ABin
            0xaa22aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000610, // AccumType, ConstantType, and PostShift
            0x5555aaaa, 0x5555aaaa, 0xaaaa5555, 0xaaaa5555,
            0x0000ffff, 0x0000ffff, 0x5555aaaa, 0x5555aaaa // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x3_nhwc2_l03_2x8 = {{
            0x00551155, // TCfg
            0x00000000, // ASelt
            0x0b0ab9a8, 0x0000dbca, // ABin
            0x00aa22aa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000610, // AccumType, ConstantType, and PostShift
            0xaaaa5555, 0xaaaa5555, 0x0000ffff, 0x0000ffff,
            0x5555aaaa, 0x5555aaaa, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};


        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l00_2x8", &uniResize_x3_nhwc2_l00_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l01_2x8", &uniResize_x3_nhwc2_l01_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l02_2x8", &uniResize_x3_nhwc2_l02_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l03_2x8", &uniResize_x3_nhwc2_l03_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l10_4x4", &uniResize_x3_nhwc2_l10_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l11_4x4", &uniResize_x3_nhwc2_l11_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l12_4x4", &uniResize_x3_nhwc2_l12_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l13_4x4", &uniResize_x3_nhwc2_l13_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l14_4x4", &uniResize_x3_nhwc2_l14_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l15_4x4", &uniResize_x3_nhwc2_l15_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l16_4x4", &uniResize_x3_nhwc2_l16_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l17_4x4", &uniResize_x3_nhwc2_l17_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_4x_up_kernel)
    {
        gpu_dp_inst_t uniResize_x4_nhwc2_l00_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x46194040, 0x1940409c, 0x48829c46, 0x82acca3a, 0xacca3a48, // BinSelect
            0x00000706, // AccumType, ConstantType, and PostShift
            0x190f0f09, 0x190f0f09, 0x23051503, 0x23051503,
            0x05230315, 0x05230315, 0x0f19090f, 0x0f19090f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x4_nhwc2_l01_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0xca3a4882, 0x3a4882ac, 0x50c4acca, 0xc4bd4e5b, 0xbd4e5b50, // BinSelect
            0x00000706, // AccumType, ConstantType, and PostShift
            0x190f0f09, 0x190f0f09, 0x23051503, 0x23051503,
            0x05230315, 0x05230315, 0x0f19090f, 0x0f19090f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x4_nhwc2_l10_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0x46194040, 0x1940409c, 0x48829c46, 0x82acca3a, 0xacca3a48, // BinSelect
            0x00000706, // AccumType, ConstantType, and PostShift
            0x23150503, 0x23150503, 0x31070701, 0x31070701,
            0x07310107, 0x07310107, 0x15230305, 0x15230305 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x4_nhwc2_l11_4x8 = {{
            0x55555555, 0x55555555, // TCfg
            0xca3a4882, 0x3a4882ac, 0x50c4acca, 0xc4bd4e5b, 0xbd4e5b50, // BinSelect
            0x00000706, // AccumType, ConstantType, and PostShift
            0x23150503, 0x23150503, 0x31070701, 0x31070701,
            0x07310107, 0x07310107, 0x15230305, 0x15230305 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize_x4_nhwc2_l00_4x8", &uniResize_x4_nhwc2_l00_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x4_nhwc2_l01_4x8", &uniResize_x4_nhwc2_l01_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x4_nhwc2_l10_4x8", &uniResize_x4_nhwc2_l10_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x4_nhwc2_l11_4x8", &uniResize_x4_nhwc2_l11_4x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        VSILOGE("input or output's format is not support");
        status = VSI_FAILURE;
        goto final;
    }

    gpu_param.global_size[0]   = gpu_align_p2((out_width  + \
        gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (out_height  + \
        gpu_param.global_scale[1] - 1) / gpu_param.global_scale[1];
    gpu_param.dim              = 2;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );

    return status;
} /* _resize_bilinear_initializer() */

DEF_KERNEL_INITIALIZER(_bilinear_nhwc_bound_initializer)
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
    vsi_nn_kernel_tensor_attr_t * output_attr = NULL;
    vsi_nn_kernel_tensor_attr_t * input_attr  = NULL;
    vsi_size_array_t             * in_shape  = NULL;
    vsi_size_array_t             * out_shape  = NULL;
    uint32_t  x_coord[2] = {0};
    uint32_t    in_width;
    uint32_t    in_height;
    uint32_t    out_width;
    uint32_t    out_height;
    vsi_bool    is_2x_up_kernel  = FALSE;
    vsi_bool    is_3x_up_kernel  = FALSE;
    vsi_bool    is_4x_up_kernel  = FALSE;


    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    in_shape = input_attr->shape;
    out_shape = output_attr->shape;

    in_width          = (uint32_t)(in_shape->data[0]);
    in_height         = (uint32_t)(in_shape->data[1]);
    out_width         = (uint32_t)(out_shape->data[0]);
    out_height        = (uint32_t)(out_shape->data[1]);

    is_2x_up_kernel = (2 * in_width == out_width) && (2 * in_height == out_height);
    is_3x_up_kernel = (3 * in_width == out_width) && (3 * in_height == out_height);
    is_4x_up_kernel = (4 * in_width == out_width) && (4 * in_height == out_height);


    if (is_2x_up_kernel)
    {
        gpu_dp_inst_t uniResize_x2_nhwc2_0_4x8 = {{
            0x55555511, 0x55555555, // TCfg
            0x46104000, 0x3a48829c, 0x4882acca, 0xc4acca3a, 0xbd4e5b50, // BinSelect
            0x00000704, // AccumType, ConstantType, and PostShift
            0x000c0004, 0x09030301, 0x03090103, 0x03090103,
            0x09030301, 0x09030301, 0x03090103, 0x03090103 // Constant
        }, GPU_DP_TYPE_16};

        gpu_param.global_scale[0] = 2;
        gpu_param.global_scale[1] = 1;
        x_coord[1] = (uint32_t)(out_shape->data[0]) - 2;
        x_coord[0] = (x_coord[1] * 2 - 1) >> 2;

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize_x2_nhwc2_0_4x8", &uniResize_x2_nhwc2_0_4x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_3x_up_kernel)
    {
        gpu_dp_inst_t uniResize_x3_nhwc2_l10_4x4 = {{
            0x05055511, // TCfg
            0x04045010, // ASelt
            0x31310000, 0x00330022, // ABin
            0x0a0aaa22, // BSelt
            0x00000000, 0x00000000, // BBin
            0x0000060f, // AccumType, ConstantType, and PostShift
            0x00005556, 0x00002aab, 0x38e41c72, 0x1c720e39,
            0x2aab5556, 0x00000000, 0x2aab5556, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_param.global_scale[0] = 3;
        gpu_param.global_scale[1] = 1;
        x_coord[1] = (uint32_t)(out_shape->data[0]) - 2;
        x_coord[0] = (x_coord[1] - 1) / 6 * 2;

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize_x3_nhwc2_l10_4x4", &uniResize_x3_nhwc2_l10_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (is_4x_up_kernel)
    {
        gpu_dp_inst_t uniResize_x4_nhwc2_l00_4x8 = {{
            0x55555511, 0x55555555, // TCfg
            0x46104000, 0x1940409c, 0x48829c46, 0x82acca3a, 0xacca3a48, // BinSelect
            0x00000706, // AccumType, ConstantType, and PostShift
            0x00280018, 0x190f0f09, 0x23051503, 0x23051503,
            0x05230315, 0x05230315, 0x0f19090f, 0x0f19090f // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniResize_x4_nhwc2_l10_4x8 = {{
            0x55555511, 0x55555555, // TCfg
            0x46104000, 0x1940409c, 0x48829c46, 0x82acca3a, 0xacca3a48, // BinSelect
            0x00000706, // AccumType, ConstantType, and PostShift
            0x00380008, 0x23150503, 0x31070701, 0x31070701,
            0x07310107, 0x07310107, 0x15230305, 0x15230305 // Constant
        }, GPU_DP_TYPE_16};


        gpu_param.global_scale[0] = 4;
        gpu_param.global_scale[1] = 1;
        x_coord[1] = (uint32_t)(out_shape->data[0]) - 2;
        x_coord[0] = ((x_coord[1] - 3) >> 3) * 2;

        status  = vsi_nn_kernel_gpu_add_param( node, "uniResize_x4_nhwc2_l00_4x8", &uniResize_x4_nhwc2_l00_4x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniResize_x4_nhwc2_l10_4x8", &uniResize_x4_nhwc2_l10_4x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        VSILOGE("input or output's format is not support");
        status = VSI_FAILURE;
        goto final;
    }

    gpu_param.global_size[0]   = gpu_align_p2((out_height  + \
        gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = 1;
    gpu_param.dim              = 2;

    status |= vsi_nn_kernel_gpu_add_param( node, "x_coord", &x_coord);
    CHECK_STATUS_FAIL_GOTO(status, final );

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );

    return status;
} /* _bilinear_nhwc_bound_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    const uint32_t hashkey,
    uint32_t kernel_id
    )
{
    vx_kernel_initialize_f  initializer = NULL;
    vx_param_description_t * param_def;
    vsi_status status = VSI_FAILURE;
    const _kernel_map_type* kernel_map;
    size_t kernel_map_size;
    size_t param_size;
    uint32_t i = 0;

    switch( kernel_id )
    {
        case 0:
            initializer = _resize_bilinear_nhwc_initializer;
            kernel_map = _resize_bilinear_nhwc_kernel_map;
            kernel_map_size = _cnt_of_array( _resize_bilinear_nhwc_kernel_map );
            param_def = _resize_bilinear_nhwc_kernel_param_def;
            param_size = _RESIZE_BILINEAR_NHWC_PARAM_NUM;
            break;
        case 1:
            initializer = _bilinear_nhwc_bound_initializer;
            kernel_map = _bilinear_nhwc_bound_kernel_map;
            kernel_map_size = _cnt_of_array( _bilinear_nhwc_bound_kernel_map );
            param_def = _bilinear_nhwc_bound_kernel_param_def;
            param_size = _BILINEAR_NHWC_BOUND_PARAM_NUM;
            break;
        default:
            VSI_ASSERT( FALSE );
            return VSI_FAILURE;
    }

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == hashkey )
        {
            break;
        }
    }
    if( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_size;
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
    vsi_nn_kernel_node_param_t node0_params[_RESIZE_BILINEAR_NHWC_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_param_t node1_params[_BILINEAR_NHWC_BOUND_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node   = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );
    vsi_bool is_same_type       = vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype);
    vsi_size_t depth            = inputs[0]->attr.size[0];
    float scale_x               = (float)outputs[0]->attr.size[1] / (float)inputs[0]->attr.size[1];
    float scale_y               = (float)outputs[0]->attr.size[2] / (float)inputs[0]->attr.size[2];
    float up_scale              = scale_x == scale_y ? scale_x : 0;
    uint32_t rank               = inputs[0]->attr.dim_num;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t  shapes[2][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_kernel_t * ikernels[2] = { NULL };
    uint32_t hashkeys[2] = {0};
    uint32_t i = 0;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;

    if (!is_same_type || depth != 2 || rank < 3 ||
        (up_scale != 2.0f && up_scale != 3.0f && up_scale != 4.0f))
    {
        return NULL;
    }

    ikernels[0] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
    // Assign unique_id
    ikernels[0]->unique_id = kernel->unique_id;
    ikernels[1] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
    // Assign unique_id
    ikernels[1]->unique_id = kernel->unique_id;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    hashkeys[0] = RESIZE_BILINEAR_NHWC_HASH_KEY( in_dtype, out_dtype, half_pixel_centers,
        align_corners, (vsi_size_t)up_scale );
    hashkeys[1] = BILINEAR_NHWC_BOUND_HASH_KEY( in_dtype, out_dtype, (vsi_size_t)up_scale );

    status = _query_kernel( ikernels[0], hashkeys[0], 0);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = _query_kernel( kernel, hashkeys[1], 1);
    CHECK_STATUS_FAIL_GOTO(status, final );

    shapes[0][0] = depth * inputs[0]->attr.size[1];
    shapes[0][1] = inputs[0]->attr.size[2];
    shapes[0][2] = 1;
    shapes[0][3] = inputs[0]->attr.size[3];

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], rank );

    shapes[1][0] = depth * outputs[0]->attr.size[1];
    shapes[1][1] = outputs[0]->attr.size[2];
    shapes[1][2] = 1;
    shapes[1][3] = outputs[0]->attr.size[3];

    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[1], rank );

    // resize bilinear
    node = vsi_nn_kernel_create_node( graph, ikernels[0] );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( node0_params, _RESIZE_BILINEAR_NHWC_PARAM_NUM,
            reshape_tensors, input_num, &reshape_tensors[1], output_num );
    node0_params[SCALAR_ALIGN_CORNERS] = vsi_nn_kernel_scalar_create( graph, I32, &align_corners );
    node0_params[SCALAR_HALF_PIXEL] = vsi_nn_kernel_scalar_create( graph, I32, &half_pixel_centers );
    status  = vsi_nn_kernel_node_pass_param( node, node0_params, _RESIZE_BILINEAR_NHWC_PARAM_NUM );
    vsi_nn_kernel_scalar_release( &node0_params[SCALAR_ALIGN_CORNERS] );
    vsi_nn_kernel_scalar_release( &node0_params[SCALAR_HALF_PIXEL] );
    vsi_nn_kernel_node_release( &node );

    // update bound for output tensor
    memcpy( &attr, &(reshape_tensors[1]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.size[0] = 1;
    attr.size[1] = 1;
    attr.dim_num = 2;
    reshape_tensors[2] = vsi_nn_CreateTensor( graph, &attr );
    node = vsi_nn_kernel_create_node( graph, kernel );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( node1_params, _BILINEAR_NHWC_BOUND_PARAM_NUM,
            reshape_tensors, 2, &reshape_tensors[2], 1 );
    status  = vsi_nn_kernel_node_pass_param( node, node1_params, _BILINEAR_NHWC_BOUND_PARAM_NUM );

final:
    for( i = 0; i < 2; i ++ )
    {
        if( ikernels[i] )
        {
            vsi_nn_kernel_release( &ikernels[i] );
        }
    }
    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);
    vsi_safe_release_tensor(reshape_tensors[2]);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( resize_bilinear_nhwc, _setup )

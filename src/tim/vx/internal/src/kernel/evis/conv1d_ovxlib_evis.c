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
#include "vsi_nn_error.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    NORMAL = 0,
    K3_S1,
    K3_S1_D2_D4,
    K1024_SMALL,
    K1024_LARGE,
} _internal_kernel_e;

#define _CONV1D_OVXLIB_KERNEL_SOURCE         "conv1d_ovxlib"
#define _CONV1D_OVXLIB_KERNEL_SOURCE_K1024   "conv1d_ovxlib_k1024"

#define STR(a) #a
// Add kernel hashtable here
#define CONV1D_OVXLIB_HASH_KEY( IN_DTYPE, W_DTYPE, B_DTYPE, OUT_DTYPE, KERNEL_TYPE ) \
        (( KERNEL_TYPE << 24 ) | ( IN_DTYPE << 18 ) | ( W_DTYPE << 12 ) | ( B_DTYPE << 6 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN_DTYPE, W_DTYPE, B_DTYPE, OUT_DTYPE, KERNEL_TYPE, SOURCE ) \
        { CONV1D_OVXLIB_HASH_KEY( IN_DTYPE, W_DTYPE, B_DTYPE, OUT_DTYPE, KERNEL_TYPE ), \
         CVIVANTE_NAMESPACE(\
         "evis.conv1d_"STR(IN_DTYPE)STR(W_DTYPE)STR(B_DTYPE)"to"STR(OUT_DTYPE)"_"STR(KERNEL_TYPE)), \
         SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _conv1d_ovxlib_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8, I32, U8, K3_S1, _CONV1D_OVXLIB_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U8, U8, I32, U8, K3_S1_D2_D4, _CONV1D_OVXLIB_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U8, U8, I32, U8, K1024_SMALL, _CONV1D_OVXLIB_KERNEL_SOURCE_K1024 ),
    PACK_KERNEL_MAP( U8, U8, I32, U8, K1024_LARGE, _CONV1D_OVXLIB_KERNEL_SOURCE_K1024 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _conv1d_ovxlib_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _CONV1D_OVXLIB_PARAM_NUM  _cnt_of_array( _conv1d_ovxlib_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_conv1d_ovxlib_initializer)
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
    vsi_nn_kernel_tensor_attr_t * input_attr    = NULL;
    vsi_nn_kernel_tensor_attr_t * weights_attr  = NULL;
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_size_array_t * in_shape                  = NULL;
    vsi_size_array_t * out_shape                 = NULL;
    vsi_size_array_t * weight_shape              = NULL;
    float             scaleIn         = 1.0f;
    float             scaleOut        = 1.0f;
    float             scaleWights     = 1.0f;
    int32_t           input_ZP        = 0;
    int32_t           weight_ZP       = 0;
    float             output_ZP       = 0;
    int32_t           stride          = 1;
    int32_t           dilation        = 0;
    int32_t           input_height    = 0;
    int32_t           input_width     = 0;
    int32_t           output_width    = 0;

    VSI_UNREFERENCED(param_size);

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );

    weights_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( weights_attr, "Create tensor attr buffer fail.", final );

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &(stride));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &(dilation));

    in_shape     = input_attr->shape;
    out_shape    = output_attr->shape;
    weight_shape = weights_attr->shape;

    input_ZP     = input_attr->zero_point;
    scaleIn      = input_attr->scale;
    weight_ZP    = weights_attr->zero_point;
    scaleWights  = weights_attr->scale;
    output_ZP    = (float)output_attr->zero_point;
    scaleOut     = output_attr->scale;

    scaleOut     = (scaleIn * scaleWights) / scaleOut;
    input_height = (int32_t)(in_shape->data[1]);
    input_width  = (int32_t)(in_shape->data[0]);
    output_width = (int32_t)(out_shape->data[0]);

    if ((U8 == input_attr->dtype) && (U8 == weights_attr->dtype) && (U8 == output_attr->dtype))
    {
        gpu_dp_inst_t uniSumOrderUchar_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x0c080400, 0x0c080400, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        if ( (3 == weight_shape->data[0]) && (1 == stride) )
        {
            gpu_dp_inst_t uniConv1DK3_Lo0_4x4 = {{
                0x69696969, // TCfg
                0x44444444, // ASelt
                0x41014000, 0x43034202, // ABin
                0x55555555, // BSelt
                0x55405540, 0x55405540, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConv1DK3_Lo1_4x4 = {{
                0x69696969, // TCfg
                0x44444444, // ASelt
                0x41114010, 0x43134212, // ABin
                0x55555555, // BSelt
                0x55415541, 0x55415541, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConv1DK3_Lo2_4x4 = {{
                0x69696969, // TCfg
                0x44444444, // ASelt
                0x41214020, 0x43234222, // ABin
                0x55555555, // BSelt
                0x55425542, 0x55425542, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConv1DK3_Hi0_4x4 = {{
                0x69696969, // TCfg
                0x44444444, // ASelt
                0x45054404, 0x47074606, // ABin
                0x55555555, // BSelt
                0x55405540, 0x55405540, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConv1DK3_Hi1_4x4 = {{
                0x69696969, // TCfg
                0x44444444, // ASelt
                0x45154414, 0x47174616, // ABin
                0x55555555, // BSelt
                0x55415541, 0x55415541, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConv1DK3_Hi2_4x4 = {{
                0x69696969, // TCfg
                0x44444444, // ASelt
                0x45254424, 0x47274626, // ABin
                0x55555555, // BSelt
                0x55425542, 0x55425542, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniDataConvK3_2x8 = {{
                0x00111111, // TCfg
                0x00110000, // ASelt
                0x03020100, 0x00000504, // ABin
                0x00222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            uint32_t conv1dK3D2_Lo1[4] = {0x43134212, 0x45154414, 0x55415541, 0x55415541};
            uint32_t conv1dK3D2_Lo2[4] = {0x45254424, 0x47274626, 0x55425542, 0x55425542};
            uint32_t conv1dK3D2_Hi1[4] = {0x47174616, 0x49194818, 0x55415541, 0x55415541};
            uint32_t conv1dK3D2_Hi2[4] = {0x49294828, 0x4b2b4a2a, 0x55425542, 0x55425542};
            uint32_t conv1dK3D4_Lo1[4] = {0x45154414, 0x47174616, 0x55415541, 0x55415541};
            uint32_t conv1dK3D4_Lo2[4] = {0x49294828, 0x4b2b4a2a, 0x55425542, 0x55425542};
            uint32_t conv1dK3D4_Hi1[4] = {0x49194818, 0x4b1b4a1a, 0x55415541, 0x55415541};
            uint32_t conv1dK3D4_Hi2[4] = {0x4d2d4c2c, 0x4f2f4e2e, 0x55425542, 0x55425542};

            if (2 == dilation)
            {
                uniConv1DK3_Lo1_4x4.data[2] = conv1dK3D2_Lo1[0];
                uniConv1DK3_Lo1_4x4.data[3] = conv1dK3D2_Lo1[1];
                uniConv1DK3_Lo1_4x4.data[5] = conv1dK3D2_Lo1[2];
                uniConv1DK3_Lo1_4x4.data[6] = conv1dK3D2_Lo1[3];
                uniConv1DK3_Lo2_4x4.data[2] = conv1dK3D2_Lo2[0];
                uniConv1DK3_Lo2_4x4.data[3] = conv1dK3D2_Lo2[1];
                uniConv1DK3_Lo2_4x4.data[5] = conv1dK3D2_Lo2[2];
                uniConv1DK3_Lo2_4x4.data[6] = conv1dK3D2_Lo2[3];
                uniConv1DK3_Hi1_4x4.data[2] = conv1dK3D2_Hi1[0];
                uniConv1DK3_Hi1_4x4.data[3] = conv1dK3D2_Hi1[1];
                uniConv1DK3_Hi1_4x4.data[5] = conv1dK3D2_Hi1[2];
                uniConv1DK3_Hi1_4x4.data[6] = conv1dK3D2_Hi1[3];
                uniConv1DK3_Hi2_4x4.data[2] = conv1dK3D2_Hi2[0];
                uniConv1DK3_Hi2_4x4.data[3] = conv1dK3D2_Hi2[1];
                uniConv1DK3_Hi2_4x4.data[5] = conv1dK3D2_Hi2[2];
                uniConv1DK3_Hi2_4x4.data[6] = conv1dK3D2_Hi2[3];
            }
            else if (4 == dilation)
            {
                uniConv1DK3_Lo1_4x4.data[2] = conv1dK3D4_Lo1[0];
                uniConv1DK3_Lo1_4x4.data[3] = conv1dK3D4_Lo1[1];
                uniConv1DK3_Lo1_4x4.data[5] = conv1dK3D4_Lo1[2];
                uniConv1DK3_Lo1_4x4.data[6] = conv1dK3D4_Lo1[3];
                uniConv1DK3_Lo2_4x4.data[2] = conv1dK3D4_Lo2[0];
                uniConv1DK3_Lo2_4x4.data[3] = conv1dK3D4_Lo2[1];
                uniConv1DK3_Lo2_4x4.data[5] = conv1dK3D4_Lo2[2];
                uniConv1DK3_Lo2_4x4.data[6] = conv1dK3D4_Lo2[3];
                uniConv1DK3_Hi1_4x4.data[2] = conv1dK3D4_Hi1[0];
                uniConv1DK3_Hi1_4x4.data[3] = conv1dK3D4_Hi1[1];
                uniConv1DK3_Hi1_4x4.data[5] = conv1dK3D4_Hi1[2];
                uniConv1DK3_Hi1_4x4.data[6] = conv1dK3D4_Hi1[3];
                uniConv1DK3_Hi2_4x4.data[2] = conv1dK3D4_Hi2[0];
                uniConv1DK3_Hi2_4x4.data[3] = conv1dK3D4_Hi2[1];
                uniConv1DK3_Hi2_4x4.data[5] = conv1dK3D4_Hi2[2];
                uniConv1DK3_Hi2_4x4.data[6] = conv1dK3D4_Hi2[3];
            }


            status  = vsi_nn_kernel_gpu_add_param( node,
                    "uniConv1DK3_Lo0_4x4", &uniConv1DK3_Lo0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConv1DK3_Hi0_4x4", &uniConv1DK3_Hi0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConv1DK3_Lo1_4x4", &uniConv1DK3_Lo1_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConv1DK3_Lo2_4x4", &uniConv1DK3_Lo2_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConv1DK3_Hi1_4x4", &uniConv1DK3_Hi1_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConv1DK3_Hi2_4x4", &uniConv1DK3_Hi2_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniDataConvK3_2x8", &uniDataConvK3_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumOrderUchar_2x8", &uniSumOrderUchar_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node, "input_ZP", &input_ZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if ( (1024 == weight_shape->data[0]) && (1 == stride) )
        {
            gpu_dp_inst_t uniU8SubZp_lo_2x8= {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8SubZp_hi_2x8= {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8Conv1d_part0_8x2= {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0x76543210, 0x87654321, // ABin
                0x55555555, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8Conv1d_part1_8x2= {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0x98765432, 0xa9876543, // ABin
                0x55555555, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8Conv1d_part2_8x2= {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0xba987654, 0xcba98765, // ABin
                0x55555555, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8Conv1d_part3_8x2= {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0xdcba9876, 0xedcba987, // ABin
                0x55555555, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            int32_t kernel_cnt_x16 = (int32_t)((weight_shape->data[0] + 15) / 16);
            status  = vsi_nn_kernel_gpu_add_param( node,
                    "kernel_cnt_x16", &kernel_cnt_x16 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8SubZp_lo_2x8", &uniU8SubZp_lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8SubZp_hi_2x8", &uniU8SubZp_hi_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8Conv1d_part0_8x2", &uniU8Conv1d_part0_8x2 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8Conv1d_part1_8x2", &uniU8Conv1d_part1_8x2 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8Conv1d_part2_8x2", &uniU8Conv1d_part2_8x2 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8Conv1d_part3_8x2", &uniU8Conv1d_part3_8x2 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniSumOrderUchar_2x8", &uniSumOrderUchar_2x8 );
            if (input_width >= GPU_TENSOR_MAX_WIDTH)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "input_width", &input_width);
                status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &output_width);
            }
            CHECK_STATUS_FAIL_GOTO(status, final );
        }

        status  = vsi_nn_kernel_gpu_add_param( node, "weight_ZP", &weight_ZP);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &output_ZP);
        status |= vsi_nn_kernel_gpu_add_param( node, "scaleOut", &scaleOut);
        status |= vsi_nn_kernel_gpu_add_param( node, "input_height", &input_height);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.dim =  2;
    gpu_param.global_size[0] = (
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0]);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input_attr);
    SAFE_FREE_TENSOR_ATTR(weights_attr);
    SAFE_FREE_TENSOR_ATTR(output_attr);

    return status;
} /* _conv1d_ovxlib_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    _internal_kernel_e   kernel_type
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e w_dtype;
    vsi_nn_kernel_dtype_e b_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _conv1d_ovxlib_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _conv1d_ovxlib_kernel_map );
    vx_param_description_t * param_def  = _conv1d_ovxlib_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _conv1d_ovxlib_kernel_param_def );
    vx_kernel_initialize_f  initializer = _conv1d_ovxlib_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    w_dtype   = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    b_dtype   = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = CONV1D_OVXLIB_HASH_KEY( in_dtype, w_dtype, b_dtype, out_dtype, kernel_type );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
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

static vsi_nn_tensor_t* _create_new_bias_tensor
    (
    vsi_nn_graph_t  *graph,
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
    )
{
    vsi_nn_tensor_t * new_bias   = NULL;
    vsi_nn_tensor_attr_t attr;
    int32_t  *new_bias_data_ptr  = NULL;
    uint8_t  *weight_data        = NULL;
    int32_t  *bias_data          = NULL;
    uint32_t  i, j;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    weight_data = vsi_nn_ConvertTensorToData(graph, weight);
    CHECK_PTR_FAIL_GOTO( weight_data, "Create buffer fail.", final );

    if (bias == NULL)
    {
        memcpy(&attr, &weight->attr, sizeof(vsi_nn_tensor_attr_t));
        attr.dim_num = 2;
        attr.size[0] = weight->attr.size[2];
        attr.size[1] = 1;
        if (weight->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            attr.dtype.scale = input->attr.dtype.scale * weight->attr.dtype.scale;
            attr.dtype.zero_point = 0;
            attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        }
    }
    else
    {
        memcpy(&attr, &bias->attr, sizeof(vsi_nn_tensor_attr_t));
        if (attr.dim_num == 1)
        {
            attr.size[1]  = 1;
            attr.dim_num  = 2;
        }
        bias_data = (int32_t *)vsi_nn_ConvertTensorToData(graph, bias);
    }

    new_bias_data_ptr = (int32_t *)malloc(attr.size[0] * sizeof(int32_t));
    CHECK_PTR_FAIL_GOTO( new_bias_data_ptr, "Create buffer fail.", final );
    memset((void *)new_bias_data_ptr, 0, sizeof(int32_t) * attr.size[0]);

    if (input->attr.dtype.zero_point != 0)
    {
        for (i = 0; i < weight->attr.size[2]; i++)
        {
            uint8_t *weight_ptr = weight_data + i * weight->attr.size[0] * weight->attr.size[1];
            for (j = 0; j < weight->attr.size[0] * weight->attr.size[1]; j++)
            {
                 new_bias_data_ptr[i] += -((int32_t)weight_ptr[j] - weight->attr.dtype.zero_point) \
                                         * input->attr.dtype.zero_point;
            }
        }
    }

    if (bias_data != NULL)
    {
        for (i = 0; i < attr.size[0]; i++)
        {
            new_bias_data_ptr[i] += bias_data[i];
        }
    }

    new_bias = vsi_nn_CreateTensorFromData(graph, (uint8_t *)new_bias_data_ptr, &attr);

final:
    vsi_nn_safe_free( new_bias_data_ptr );
    vsi_nn_safe_free( bias_data );
    vsi_nn_safe_free( weight_data );

    return new_bias;
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
    vsi_nn_kernel_node_param_t node_params[_CONV1D_OVXLIB_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t j = 0;
    _internal_kernel_e      kernel_type = NORMAL;

    int32_t stride          = vsi_nn_kernel_param_get_int32( params, "stride" );
    int32_t pad_front       = vsi_nn_kernel_param_get_int32( params, "pad_front" );
    int32_t pad_end         = vsi_nn_kernel_param_get_int32( params, "pad_end" );
    int32_t dilation        = vsi_nn_kernel_param_get_int32( params, "dilation" );
    int32_t overflow_policy = vsi_nn_kernel_param_get_int32( params, "overflow_policy" );
    vsi_nn_tensor_t *in_tensors[3] = {NULL};
    vsi_nn_tensor_t *new_bias = NULL;

    if (VX_CONVERT_POLICY_SATURATE == overflow_policy)
    {
        overflow_policy = 1;
    }
    else
    {
        overflow_policy = 0;
    }

    if ( 1 == stride )
    {
        if ( 3 == inputs[1]->attr.size[0] )
        {
            if (2 == dilation || 4 == dilation)
            {
                 kernel_type = K3_S1_D2_D4;
            }
            else
            {
                kernel_type = K3_S1;
            }
        }
        else if ( 1024 == inputs[1]->attr.size[0] )
        {
            if (inputs[0]->attr.size[0] < 65535)
            {
                kernel_type = K1024_SMALL;
            }
            else if (0 == pad_front && 0 == pad_end)
            {
                kernel_type = K1024_LARGE;
            }
            else
            {
                return NULL;
            }
        }
        else
        {
            return NULL;
        }
    }

    if (1024 == inputs[1]->attr.size[0])
    {
        new_bias = _create_new_bias_tensor(graph, inputs[0], inputs[1], inputs[2]);
        in_tensors[0] = inputs[0];
        in_tensors[1] = inputs[1];
        in_tensors[2] = new_bias;
    }
    else
    {
        in_tensors[0] = inputs[0];
        in_tensors[1] = inputs[1];
        in_tensors[2] = inputs[2];
    }

    status = _query_kernel( kernel, inputs, outputs, kernel_type );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            if( pad_front != 0 || pad_end != 0)
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = (uint8_t)(inputs[0]->attr.dtype.zero_point);
                status |= vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            }

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CONV1D_OVXLIB_PARAM_NUM,
                    in_tensors, input_num, outputs, output_num );
            j = (int32_t)(input_num + output_num);
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &stride );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &pad_front );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &pad_end );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &dilation );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &overflow_policy );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CONV1D_OVXLIB_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
        }
    }

    if (new_bias)
    {
        vsi_nn_ReleaseTensor(&new_bias);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( conv1d_ovxlib, _setup )

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
    KN = 0,
    K40,
    K56,
    K64,
    K80,
    K88,
} _internal_kernel_size_e;


typedef enum
{
    D0 = 0,
    D1,
    D2,
} _internal_dilation_e;

typedef enum
{
    PARAM_INPUT = 0,
    PARAM_WEIGHT,
    PARAM_BIAS,
    PARAM_OUTPUT,
    PARAM_PAD,
    PARAM_STRIDE,
    PARAM_DILATION,
} param_index_e;

#define _DEPTHWISE_CONV1D_KERNEL_SOURCE0      "depthwise_conv1d_src0"
#define _DEPTHWISE_CONV1D_KERNEL_SOURCE1      "depthwise_conv1d_src1"
#define _DEPTHWISE_CONV1D_KERNEL_SOURCE2      "depthwise_conv1d_src2"
#define _DEPTHWISE_CONV1D_KERNEL_SOURCE3      "depthwise_conv1d_src3"

#define STR(a) #a

// Add kernel hashtable here
#define DEPTHWISE_CONV1D_HASH_KEY( SRC_TYPE, WEIGHT_TYPE, DST_TYPE, KERNEL_SIZE, DILATION ) \
        ((DILATION << 23) | (KERNEL_SIZE << 15) | (WEIGHT_TYPE << 10) | ( SRC_TYPE << 5 ) | ( DST_TYPE ))

#define PACK_KERNEL_MAP( SRC_TYPE, DST_TYPE, WEIGHT_TYPE, KERNEL_SIZE, DILATION, SOURCE) \
        { DEPTHWISE_CONV1D_HASH_KEY( SRC_TYPE, WEIGHT_TYPE, DST_TYPE, KERNEL_SIZE, DILATION ),\
          CVIVANTE_NAMESPACE("evis.vxDW_Conv1D_"STR(SRC_TYPE)"to"STR(DST_TYPE)"_"STR(KERNEL_SIZE)"_"STR(DILATION)) \
          , SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _depthwise_conv1d_kernel_map[] =
{
    PACK_KERNEL_MAP( U8, U8, U8, KN,  D1, _DEPTHWISE_CONV1D_KERNEL_SOURCE0 ),
    PACK_KERNEL_MAP( U8, U8, U8, KN,  D2, _DEPTHWISE_CONV1D_KERNEL_SOURCE0 ),
    PACK_KERNEL_MAP( U8, U8, U8, K40, D1, _DEPTHWISE_CONV1D_KERNEL_SOURCE1 ),
    PACK_KERNEL_MAP( U8, U8, U8, K56, D1, _DEPTHWISE_CONV1D_KERNEL_SOURCE1 ),
    PACK_KERNEL_MAP( U8, U8, U8, K64, D1, _DEPTHWISE_CONV1D_KERNEL_SOURCE2 ),
    PACK_KERNEL_MAP( U8, U8, U8, K80, D1, _DEPTHWISE_CONV1D_KERNEL_SOURCE2 ),
    PACK_KERNEL_MAP( U8, U8, U8, K88, D2, _DEPTHWISE_CONV1D_KERNEL_SOURCE3 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _depthwise_conv1d_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _DEPTHWISE_CONV1D_PARAM_NUM  _cnt_of_array( _depthwise_conv1d_kernel_param_def )

static _internal_kernel_size_e get_kernel_size(uint32_t k_size, uint32_t dilation,
                                               uint32_t stride, uint32_t evis_version)
{
#define _PACK_SELECT_KEY( kernel_size, dilation, stride, evis_version )    \
        ( (uint64_t)kernel_size | ((uint64_t)dilation << 16) \
      | ( (uint64_t)stride << 32) | ((uint64_t)evis_version << 48))

    _internal_kernel_size_e ks_e = KN;
    uint64_t pack_key = _PACK_SELECT_KEY(k_size, dilation, stride, evis_version);

    switch (pack_key)
    {
    case _PACK_SELECT_KEY(40, D1, 1, VSI_NN_HW_EVIS_2):
    case _PACK_SELECT_KEY(40, D1, 2, VSI_NN_HW_EVIS_2):
        ks_e = K40;
        break;
    case _PACK_SELECT_KEY(56, D1, 1, VSI_NN_HW_EVIS_2):
        ks_e = K56;
        break;
    case _PACK_SELECT_KEY(64, D1, 1, VSI_NN_HW_EVIS_2):
        ks_e = K64;
        break;
    case _PACK_SELECT_KEY(80, D1, 1, VSI_NN_HW_EVIS_2):
        ks_e = K80;
        break;
    case _PACK_SELECT_KEY(88, D2, 1, VSI_NN_HW_EVIS_2):
        ks_e = K88;
        break;
    default:
        ks_e = KN;
        break;
    }

#undef _PACK_SELECT_KEY

    return ks_e;
}

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_depthwise_conv1d_initializer)
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
    vx_tensor                   input           = (vx_tensor)param[0];
    vx_tensor                   weight          = (vx_tensor)param[1];
    vx_tensor                   output          = (vx_tensor)param[3];
    int32_t                     stride          = 0;
    int32_t                     dilation        = 0;
    vsi_nn_kernel_tensor_attr_t *input_attr     = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr    = NULL;
    vsi_nn_kernel_tensor_attr_t *weight_attr    = NULL;
    vsi_int_array_t             *output_shape   = NULL;
    int32_t                     weightZP        = 0;
    float                       outputScale     = 1.0f;
    float                       outputZP        = 0;
    vx_hardware_caps_params_t   hw_param;
    _internal_kernel_size_e     ks              = KN;
    uint32_t                    kernel_size     = 0;
    uint32_t                    kernel_size_x16 = 0;
    uint32_t                    kernel_size_x8  = 0;
    uint32_t                    evis_version    = 0;
    vx_context                  ctx             = vxGetContext((vx_reference)node);
    uint64_t                    pack_key        = 0;

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_t));
    status = vxQueryHardwareCaps(ctx, &hw_param, sizeof(vx_hardware_caps_params_t));
    CHECK_STATUS_FAIL_GOTO(status, final);

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input);
    CHECK_PTR_FAIL_GOTO( input_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    weight_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)weight);
    CHECK_PTR_FAIL_GOTO( weight_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &stride);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &dilation);
    CHECK_STATUS_FAIL_GOTO(status, final );
    kernel_size = weight_attr->shape->data[0];

    if(hw_param.evis1 == TRUE && hw_param.evis2 == FALSE)
    {
        evis_version = VSI_NN_HW_EVIS_1;
    }
    else if(hw_param.evis1 == FALSE && hw_param.evis2 == TRUE)
    {
        evis_version = VSI_NN_HW_EVIS_2;
    }

    ks = get_kernel_size(kernel_size, dilation, stride, evis_version);

    output_shape = output_attr->shape;
    gpu_param.dim = 2;
    gpu_param.global_offset[0] = 0;
    gpu_param.global_offset[1] = 0;

    if (KN == ks)
    {
        gpu_param.global_scale[0]  = 1;
        gpu_param.global_scale[1]  = 1;
    }
    else
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
    }

    gpu_param.local_size[0]    = 8;
    gpu_param.local_size[1]    = 1;
    gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1)
                                             / gpu_param.global_scale[0], gpu_param.local_size[0]);
    gpu_param.global_size[1]   = gpu_align_p2((output_shape->data[1] + gpu_param.global_scale[1] - 1)
                                             / gpu_param.global_scale[1], gpu_param.local_size[1]);

    outputScale = input_attr->asymm.scale;

    outputScale *= weight_attr->asymm.scale;
    weightZP     = weight_attr->asymm.zero_point;
    outputScale /= output_attr->asymm.scale;
    outputZP = (float)output_attr->asymm.zero_point + 0.5f;

#define _PACK_SELECT_KEY( kernel_size, dilation, evis_version )    \
        ((uint64_t)kernel_size | ((uint64_t)dilation << 16) | ((uint64_t)evis_version << 32))

    pack_key = _PACK_SELECT_KEY(ks, dilation, evis_version);

    switch (pack_key)
    {
    case _PACK_SELECT_KEY(KN, D1, VSI_NN_HW_EVIS_1):
    case _PACK_SELECT_KEY(KN, D2, VSI_NN_HW_EVIS_1):
    case _PACK_SELECT_KEY(KN, D1, VSI_NN_HW_EVIS_2):
    case _PACK_SELECT_KEY(KN, D2, VSI_NN_HW_EVIS_2):
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
            gpu_dp_inst_t uniU8ConvS16_align8_step0_16x1 = {{
                0x00005555, // TCfg
                0x00000000, // ASelt
                0x76543210, 0x00000000, // ABin
                0x00005555, // BSelt
                0x76543210, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_align8_step1_16x1 = {{
                0x00005555, // TCfg
                0x00000000, // ASelt
                0xfedcba98, 0x00000000, // ABin
                0x00005555, // BSelt
                0x76543210, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            gpu_dp_inst_t uniU8ConvS16_align8_dial_step_16x1 = {{
                0x00005555, // TCfg
                0x00000000, // ASelt
                0xeca86420, 0x00000000, // ABin
                0x00005555, // BSelt
                0x76543210, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            if (1 == dilation)
            {
                kernel_size_x16 = (uint32_t)(kernel_size / 16) * 16;
                kernel_size_x8  = kernel_size - kernel_size_x16;
                status  = vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_align8_step0_16x1",
                &uniU8ConvS16_align8_step0_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_align8_step1_16x1",
                &uniU8ConvS16_align8_step1_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "kernel_size_x16", &kernel_size_x16);
                status |= vsi_nn_kernel_gpu_add_param(node, "kernel_size_x8", &kernel_size_x8);
            }
            else if (2 == dilation)
            {
                kernel_size_x8  = (uint32_t)(kernel_size / 8) * 8;
                status  = vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_align8_step0_16x1",
                &uniU8ConvS16_align8_dial_step_16x1);
                status |= vsi_nn_kernel_gpu_add_param(node, "kernel_size_x8", &kernel_size_x8);
            }
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZp_lo_2x8", &uniU8SubZp_lo_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZp_hi_2x8", &uniU8SubZp_hi_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "weightZP", &weightZP);
            status |= vsi_nn_kernel_gpu_add_param(node, "scale", &outputScale);
            status |= vsi_nn_kernel_gpu_add_param(node, "outputZP", &outputZP);

            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case _PACK_SELECT_KEY(K40, D1, VSI_NN_HW_EVIS_2):
    case _PACK_SELECT_KEY(K56, D1, VSI_NN_HW_EVIS_2):
    case _PACK_SELECT_KEY(K64, D1, VSI_NN_HW_EVIS_2):
    case _PACK_SELECT_KEY(K80, D1, VSI_NN_HW_EVIS_2):
        {
            gpu_dp_inst_t uniU8ConvS16_Stpe0_8x2b= {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0x76543210, 0x98765432, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe1_8x2b= {{
                0x55555555, // TCfg
                0x00000000, // ASelt
                0xba987654, 0xdcba9876, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe2_8x2b= {{
                0x55555555, // TCfg
                0x50000000, // ASelt
                0xfedcba98, 0x10fedcba, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe3_8x2b= {{
                0x55555555, // TCfg
                0x55505500, // ASelt
                0x3210fedc, 0x543210fe, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe4_8x2b= {{
                0x55555555, // TCfg
                0x50000000, // ASelt
                0xfedcba98, 0x10fedcba, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe5_8x2b= {{
                0x55555555, // TCfg
                0x55505500, // ASelt
                0x3210fedc, 0x543210fe, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            gpu_dp_inst_t uniU8ConvS16_Stpe6_8x2b= {{
                0x55555555, // TCfg
                0x55555555, // ASelt
                0x76543210, 0x98765432, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe7_8x2b= {{
                0x55555555, // TCfg
                0x55555555, // ASelt
                0xba987654, 0xdcba9876, // ABin
                0x00000000, // BSelt
                0x76543210, 0x76543210, // BBin
                0x00000000, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

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
            gpu_dp_inst_t uniExtractInteger_2x8= {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            if (stride == 1)
            {
                uniU8ConvS16_Stpe0_8x2b.data[3] = 0x87654321;
                uniU8ConvS16_Stpe1_8x2b.data[2] = 0x98765432;
                uniU8ConvS16_Stpe1_8x2b.data[3] = 0xa9876543;
                uniU8ConvS16_Stpe2_8x2b.data[1] = 0x40000000;
                uniU8ConvS16_Stpe2_8x2b.data[3] = 0x0fedcba9;
                uniU8ConvS16_Stpe3_8x2b.data[1] = 0x54005000;
                uniU8ConvS16_Stpe3_8x2b.data[2] = 0x10fedcba;
                uniU8ConvS16_Stpe3_8x2b.data[3] = 0x210fedcb;
                uniU8ConvS16_Stpe4_8x2b.data[1] = 0x00000000;
                uniU8ConvS16_Stpe4_8x2b.data[2] = 0xba987654;
                uniU8ConvS16_Stpe4_8x2b.data[3] = 0xcba98765;
                uniU8ConvS16_Stpe5_8x2b.data[1] = 0x00000000;
                uniU8ConvS16_Stpe5_8x2b.data[2] = 0xdcba9876;
                uniU8ConvS16_Stpe5_8x2b.data[3] = 0xedcba987;
                uniU8ConvS16_Stpe6_8x2b.data[1] = 0x55405500;
                uniU8ConvS16_Stpe6_8x2b.data[2] = 0x3210fedc;
                uniU8ConvS16_Stpe6_8x2b.data[3] = 0x43210fed;
                uniU8ConvS16_Stpe7_8x2b.data[1] = 0x55545550;
                uniU8ConvS16_Stpe7_8x2b.data[2] = 0x543210fe;
                uniU8ConvS16_Stpe7_8x2b.data[3] = 0x6543210f;
            }
            status  = vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe0_8x2b", &uniU8ConvS16_Stpe0_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe1_8x2b", &uniU8ConvS16_Stpe1_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe2_8x2b", &uniU8ConvS16_Stpe2_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe3_8x2b", &uniU8ConvS16_Stpe3_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZp_lo_2x8", &uniU8SubZp_lo_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZp_hi_2x8", &uniU8SubZp_hi_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractInteger_2x8", &uniExtractInteger_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe4_8x2b", &uniU8ConvS16_Stpe4_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe5_8x2b", &uniU8ConvS16_Stpe5_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe6_8x2b", &uniU8ConvS16_Stpe6_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe7_8x2b", &uniU8ConvS16_Stpe7_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "weightZP", &weightZP);
            status |= vsi_nn_kernel_gpu_add_param(node, "scale", &outputScale);
            status |= vsi_nn_kernel_gpu_add_param(node, "outputZP", &outputZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case _PACK_SELECT_KEY(K88, D2, VSI_NN_HW_EVIS_2):
        {
            gpu_dp_inst_t uniExtractInteger_2x8= {{
                0x33333333, // TCfg
                    0x11110000, // ASelt
                    0x03020100, 0x03020100, // ABin
                    0x00000000, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00002400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe0_8x2b= {{
                0x55555555, // TCfg
                    0x00000000, // ASelt
                    0xeca86420, 0xfdb97531, // ABin
                    0x00000000, // BSelt
                    0x76543210, 0x76543210, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe1_8x2b= {{
                0x55555555, // TCfg
                    0x40004000, // ASelt
                    0x0eca8642, 0x1fdb9753, // ABin
                    0x00000000, // BSelt
                    0x76543210, 0x76543210, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe4_8x2b= {{
                0x55555555, // TCfg
                    0x50005000, // ASelt
                    0x20eca864, 0x31fdb975, // ABin
                    0x00000000, // BSelt
                    0x76543210, 0x76543210, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8ConvS16_Stpe5_8x2b= {{
                0x55555555, // TCfg
                    0x54005400, // ASelt
                    0x420eca86, 0x531fdb97, // ABin
                    0x00000000, // BSelt
                    0x76543210, 0x76543210, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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

            status  = vsi_nn_kernel_gpu_add_param(node, "uniExtractInteger_2x8", &uniExtractInteger_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe0_8x2b", &uniU8ConvS16_Stpe0_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe1_8x2b", &uniU8ConvS16_Stpe1_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe4_8x2b", &uniU8ConvS16_Stpe4_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8ConvS16_Stpe5_8x2b", &uniU8ConvS16_Stpe5_8x2b);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZp_hi_2x8", &uniU8SubZp_hi_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniU8SubZp_lo_2x8", &uniU8SubZp_lo_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "weightZP", &weightZP);
            status |= vsi_nn_kernel_gpu_add_param(node, "scale", &outputScale);
            status |= vsi_nn_kernel_gpu_add_param(node, "outputZP", &outputZP);
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        VSILOGE("unsupport kernel size:%d/dilation:%d/evis version:%d", kernel_size, dilation, evis_version);
        break;
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (input_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input_attr);
    }
    if (weight_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&weight_attr);
    }
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }

    return status;
} /* _depthwise_conv1d_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t dilation,
    _internal_kernel_size_e kernel_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e weight_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _depthwise_conv1d_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _depthwise_conv1d_kernel_map );
    vx_param_description_t * param_def  = _depthwise_conv1d_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _depthwise_conv1d_kernel_param_def );
    vx_kernel_initialize_f  initializer = _depthwise_conv1d_initializer;
    _internal_dilation_e    dilation_e = D0;
    uint32_t key = 0;
    size_t i = 0;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    weight_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    switch (dilation)
    {
    case 1:
        dilation_e = D1;
        break;
    case 2:
        dilation_e = D2;
        break;
    default:
        dilation_e = D0;
        break;
    }

    key = DEPTHWISE_CONV1D_HASH_KEY( in_dtype, weight_dtype, out_dtype, kernel_size, dilation_e);

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_DEPTHWISE_CONV1D_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t weight_pad_front[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t weight_pad_end[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_nn_tensor_t * weights = NULL;
    vsi_nn_tensor_t * biases = NULL;
    vsi_nn_tensor_t *temp_tensor[3] = {NULL};
    int32_t stride     = vsi_nn_kernel_param_get_int32( params, "stride" );
    int32_t pad_front  = vsi_nn_kernel_param_get_int32( params, "pad_front" );
    int32_t pad_end  = vsi_nn_kernel_param_get_int32( params, "pad_end" );
    int32_t dilation   = vsi_nn_kernel_param_get_int32( params, "dilation" );
    _internal_kernel_size_e ks   = KN;

    weight_pad_end[0] = gpu_align_np2_safe(inputs[1]->attr.size[0], 8) - inputs[1]->attr.size[0];

    weights = vsi_nn_pad_tensor(graph, inputs[1], weight_pad_front, weight_pad_end, inputs[1]->attr.dim_num,
        VSI_NN_PAD_MODE_CONSTANT, 0);

    biases = vsi_nn_merge_input_zeropoint_to_bias(graph, inputs[0], inputs[1], inputs[2]);

    temp_tensor[0] = inputs[0];
    temp_tensor[1] = weights;
    temp_tensor[2] = biases;

    ks = get_kernel_size(weights->attr.size[0], dilation, stride, graph->ctx->config.evis.ver);

    status = _query_kernel( kernel, temp_tensor, outputs, dilation, ks);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            if( pad_front != 0 && pad_end != 0)
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = 0;
                border.constant_value.U16 = 0;
                status |= vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            }


            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _DEPTHWISE_CONV1D_PARAM_NUM,
                    temp_tensor, input_num, outputs, output_num );
            node_params[PARAM_PAD] = vsi_nn_kernel_scalar_create( graph, I32, &pad_front );
            node_params[PARAM_STRIDE] = vsi_nn_kernel_scalar_create( graph, I32, &stride );
            node_params[PARAM_DILATION] = vsi_nn_kernel_scalar_create( graph, I32, &dilation );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _DEPTHWISE_CONV1D_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[PARAM_PAD] );
            vsi_nn_kernel_scalar_release( &node_params[PARAM_STRIDE] );
            vsi_nn_kernel_scalar_release( &node_params[PARAM_DILATION] );
        }
    }

    if (weights)
    {
        vsi_nn_ReleaseTensor(&weights);
    }

    if (biases)
    {
        vsi_nn_ReleaseTensor(&biases);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( depthwise_conv1d, _setup )


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

__BEGIN_DECLS

#define RGB888_SEP_SOURCE_0     "pre_process_rgb888_planar_sep_nhwc_0",
#define RGB888_SEP_SOURCE_1     "pre_process_rgb888_planar_sep_nhwc_1",
#define RGB888_SEP_SOURCE_2     "pre_process_rgb888_planar_sep_nhwc_2",
#define RGB888_SOURCE_0         "pre_process_rgb888_planar_nhwc_0",
#define RGB888_SOURCE_1         "pre_process_rgb888_planar_nhwc_1",
#define RGB888_SOURCE_2         "pre_process_rgb888_planar_nhwc_2",

#define STR(a) #a

typedef enum
{
    COPY = 0,
    SCALE,
    FOUR_OVER_THREE,
    HALF
} _internal_scale_e;

// Add kernel hashtable here
#define PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, SEP, SCALE_FLAG ) \
        (( IN_DTYPE << 16 ) | ( OUT_DTYPE << 8 ) | ( SEP << 4 ) | (SCALE_FLAG))

#define PACK_KERNEL_SCALE_MAP( IN_DTYPE, OUT_DTYPE ) \
   { PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0, SCALE ), \
     CVIVANTE_NAMESPACE("evis.pre_process_rgb888_planar_scale_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_nhwc"), \
     RGB888_SOURCE_0 }

#define PACK_KERNEL_SEP_SCALE_MAP( IN_DTYPE, OUT_DTYPE ) \
  { PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1, SCALE ), \
    CVIVANTE_NAMESPACE("evis.pre_process_rgb888_planar_sep_scale_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_nhwc"), \
    RGB888_SEP_SOURCE_0 }

#define PACK_KERNEL_COPY_MAP( IN_DTYPE, OUT_DTYPE ) \
   { PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0, COPY ), \
     CVIVANTE_NAMESPACE("evis.pre_process_rgb888_planar_copy_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_nhwc"), \
     RGB888_SOURCE_1 }

#define PACK_KERNEL_SEP_COPY_MAP( IN_DTYPE, OUT_DTYPE ) \
   { PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1, COPY ), \
     CVIVANTE_NAMESPACE("evis.pre_process_rgb888_planar_sep_copy_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_nhwc"), \
     RGB888_SEP_SOURCE_1 }

#define PACK_KERNEL_HALF_MAP( IN_DTYPE, OUT_DTYPE ) \
   { PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0, HALF ), \
     CVIVANTE_NAMESPACE("evis.pre_process_rgb888_planar_half_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_nhwc"), \
     RGB888_SOURCE_2 }

#define PACK_KERNEL_SEP_HALF_MAP( IN_DTYPE, OUT_DTYPE ) \
   { PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1, HALF ), \
     CVIVANTE_NAMESPACE("evis.pre_process_rgb888_planar_sep_half_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_nhwc"), \
     RGB888_SEP_SOURCE_2 }
typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _pre_process_rgb888_planar_nhwc_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_SCALE_MAP( U8, F16 ),
    PACK_KERNEL_SCALE_MAP( U8, I16 ),
    PACK_KERNEL_SCALE_MAP( U8, I8 ),
    PACK_KERNEL_SCALE_MAP( U8, U8 ),

    PACK_KERNEL_COPY_MAP( U8, F16 ),
    PACK_KERNEL_COPY_MAP( U8, I16 ),
    PACK_KERNEL_COPY_MAP( U8, I8 ),
    PACK_KERNEL_COPY_MAP( U8, U8 ),

    PACK_KERNEL_HALF_MAP( U8, U8 ),

    PACK_KERNEL_SEP_SCALE_MAP( U8, F16 ),
    PACK_KERNEL_SEP_SCALE_MAP( U8, I16 ),
    PACK_KERNEL_SEP_SCALE_MAP( U8, I8 ),
    PACK_KERNEL_SEP_SCALE_MAP( U8, U8 ),

    PACK_KERNEL_SEP_COPY_MAP( U8, F16 ),
    PACK_KERNEL_SEP_COPY_MAP( U8, I16 ),
    PACK_KERNEL_SEP_COPY_MAP( U8, I8 ),
    PACK_KERNEL_SEP_COPY_MAP( U8, U8 ),

    PACK_KERNEL_SEP_HALF_MAP( U8, U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _pre_process_rgb888_planar_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _PRE_PROCESS_RGB888_PLANAR_PARAM_NUM  _cnt_of_array( _pre_process_rgb888_planar_kernel_param_def )

static vx_param_description_t _pre_process_rgb888_planar_sep_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _PRE_PROCESS_RGB888_PLANAR_SEP_PARAM_NUM  _cnt_of_array( _pre_process_rgb888_planar_sep_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_pre_process_rgb888_planar_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float    output_zp    = 0;
    float    output_scale = 1;
    int32_t reverse = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    if (param_size == _cnt_of_array( _pre_process_rgb888_planar_sep_kernel_param_def ))
    {
        attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    }
    else
    {
        attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    }
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[param_size - 3], &reverse);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    out_shape  = attr[0]->shape;
    width = (uint32_t)(out_shape->data[0] / 3);
    height = (uint32_t)(out_shape->data[1]);
    output_scale /= attr[0]->scale;
    output_zp = (float)attr[0]->zero_point;

    shaderParam.global_scale[0]  = 4;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = height;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniVecShift10 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000400, 0x00000000, 0x00000400, 0x00000000,
            0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAddRShift = {{
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002405, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGetTempVal = {{
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x05050505, // BSelt
            0x00110000, 0x00330022, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractBytes = {{
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002414, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertIntergetoF32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveRGB_0_2x8 = {{
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x01000400, 0x06020105, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveRGB_1_2x8 = {{
            0x00001111, // TCfg
            0x00001001, // ASelt
            0x03070302, 0x00000000, // ABin
            0x00002222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveBGR_0_2x8 = {{
            0x11111111, // TCfg
            0x01001001, // ASelt
            0x01000400, 0x06020105, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveBGR_1_2x8 = {{
            0x00001111, // TCfg
            0x00000010, // ASelt
            0x03070302, 0x00000000, // ABin
            0x00002222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BitsDataInterleaveRGB_0_2x8= {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x01080400, 0x06020905, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BitsDataInterleaveRGB_1_2x8 = {{
            0x00001111, // TCfg
            0x00000000, // ASelt
            0x0b07030a, 0x00000000, // ABin
            0x00002222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BitsDataInterleaveBGR_0_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x09000408, 0x060a0105, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BitsDataInterleaveBGR_1_2x8 = {{
            0x00001111, // TCfg
            0x00000000, // ASelt
            0x03070b02, 0x00000000, // ABin
            0x00002222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniVecShift10", &uniVecShift10);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniAddRShift", &uniAddRShift);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniGetTempVal", &uniGetTempVal);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractBytes", &uniExtractBytes);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertIntergetoF32_4x4", &uniConvertIntergetoF32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
        status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
        if (reverse)
        {
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_0_2x8",
                &uni16BitsDataInterleaveBGR_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_1_2x8",
                &uni16BitsDataInterleaveBGR_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_0_2x8",
                &uni8BitsDataInterleaveBGR_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_1_2x8",
                &uni8BitsDataInterleaveBGR_1_2x8);
        }
        else
        {
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_0_2x8",
                &uni16BitsDataInterleaveRGB_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_1_2x8",
                &uni16BitsDataInterleaveRGB_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_0_2x8",
                &uni8BitsDataInterleaveRGB_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_1_2x8",
                &uni8BitsDataInterleaveRGB_1_2x8);
        }

        if (attr[0]->dtype == F16)
        {
            status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniExtractHalf8_2x8);
        }
        else
        {
            status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniExtractInteger_2x8);
        }
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _pre_process_rgb888_planar_initializer() */

DEF_KERNEL_INITIALIZER(_pre_process_rgb888_planar_copy_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float    output_zp    = 0;
    float    output_scale = 1;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t reverse = 0;

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    if (param_size == _cnt_of_array( _pre_process_rgb888_planar_sep_kernel_param_def ))
    {
        attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    }
    else
    {
        attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    }
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[param_size - 3], &reverse);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    out_shape  = attr[0]->shape;
    width = (uint32_t)(out_shape->data[0] / 3);
    height = (uint32_t)(out_shape->data[1]);

    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if ( attr[0]->dfp.fl > 0 )
        {
            output_scale *= (float)((int64_t)1 << attr[0]->dfp.fl);
        }
        else
        {
            output_scale *= (1.0f / (float)((int64_t)1 << -attr[0]->dfp.fl));
        }
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        output_zp = (float)attr[0]->asymm.zero_point;
        output_scale /= attr[0]->asymm.scale;
    }

    if (attr[0]->dtype == F16 || attr[0]->dtype == I16)
    {
        shaderParam.global_scale[0] = 4;
    }
    else
    {
        shaderParam.global_scale[0] = 8;
    }
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = height;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniDataMeanStddevLo_2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveRGB_0_2x8 = {{
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x01000400, 0x06020105, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveRGB_1_2x8 = {{
            0x00001111, // TCfg
            0x00001001, // ASelt
            0x03070302, 0x00000000, // ABin
            0x00002222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveBGR_0_2x8 = {{
            0x11111111, // TCfg
            0x01001001, // ASelt
            0x01000400, 0x06020105, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni16BitsDataInterleaveBGR_1_2x8 = {{
            0x00001111, // TCfg
            0x00000010, // ASelt
            0x03070302, 0x00000000, // ABin
            0x00002222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveRGB_0_2x8 = {{
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x01000800, 0x0a020109, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveRGB_1_2x8 = {{
            0x11111111, // TCfg
            0x01001001, // ASelt
            0x030b0302, 0x05040c04, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveRGB_2_2x8 = {{
            0x11111111, // TCfg
            0x10010010, // ASelt
            0x0e06050d, 0x070f0706, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveBGR_0_2x8 = {{
            0x11111111, // TCfg
            0x01001001, // ASelt
            0x01000800, 0x0a020109, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveBGR_1_2x8 = {{
            0x11111111, // TCfg
            0x10010010, // ASelt
            0x030b0302, 0x05040c04, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveBGR_2_2x8 = {{
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x0e06050d, 0x070f0706, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniDataMeanStddevLo_2x8", &uniDataMeanStddevLo_2x8);
        if (reverse)
        {
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_0_2x8",
                &uni16BitsDataInterleaveBGR_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_1_2x8",
                &uni16BitsDataInterleaveBGR_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_0_2x8",
                &uni8BDataInterleaveBGR_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_1_2x8",
                &uni8BDataInterleaveBGR_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_2_2x8",
                &uni8BDataInterleaveBGR_2_2x8);
        }
        else
        {
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_0_2x8",
                &uni16BitsDataInterleaveRGB_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni16BitsDataInterleave_1_2x8",
                &uni16BitsDataInterleaveRGB_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_0_2x8",
                &uni8BDataInterleaveRGB_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_1_2x8",
                &uni8BDataInterleaveRGB_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_2_2x8",
                &uni8BDataInterleaveRGB_2_2x8);
        }
        status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _pre_process_gray_copy_initializer() */

DEF_KERNEL_INITIALIZER(_resize_rgb888_planar_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    int32_t reverse = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    if (param_size == _cnt_of_array( _pre_process_rgb888_planar_sep_kernel_param_def ))
    {
        attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    }
    else
    {
        attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    }
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[param_size - 3], &reverse);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    {
        shaderParam.global_scale[0]  = 16;
        shaderParam.global_scale[1]  = 2;
        shaderParam.global_size[0]   = gpu_align_p2((attr[0]->shape->data[0] + shaderParam.global_scale[0] - 1)
            / shaderParam.global_scale[0], 4);
        shaderParam.global_size[1]   = (attr[0]->shape->data[1] + shaderParam.global_scale[1] - 1)
            / shaderParam.global_scale[1];
    }

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uni8BDataInterleaveRGB_0_2x8 = {{
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x01000800, 0x0a020109, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveRGB_1_2x8 = {{
            0x11111111, // TCfg
            0x01001001, // ASelt
            0x030b0302, 0x05040c04, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveRGB_2_2x8 = {{
            0x11111111, // TCfg
            0x10010010, // ASelt
            0x0e06050d, 0x070f0706, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveBGR_0_2x8 = {{
            0x11111111, // TCfg
            0x01001001, // ASelt
            0x01000800, 0x0a020109, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveBGR_1_2x8 = {{
            0x11111111, // TCfg
            0x10010010, // ASelt
            0x030b0302, 0x05040c04, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uni8BDataInterleaveBGR_2_2x8 = {{
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x0e06050d, 0x070f0706, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        if (reverse)
        {
            status = vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_0_2x8",
                &uni8BDataInterleaveBGR_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_1_2x8",
                &uni8BDataInterleaveBGR_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_2_2x8",
                &uni8BDataInterleaveBGR_2_2x8);
        }
        else
        {
            status  = vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_0_2x8",
                &uni8BDataInterleaveRGB_0_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_1_2x8",
                &uni8BDataInterleaveRGB_1_2x8);
            status |= vsi_nn_kernel_gpu_add_param(node, "uni8BitsDataInterleave_2_2x8",
                &uni8BDataInterleaveRGB_2_2x8);
        }
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }

    return status;
} /* _resize_rgb888_planar_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_nn_kernel_t* kernel,
    const vsi_nn_kernel_param_t * params,
    vsi_bool is_no_range_change,
    int32_t width,
    int32_t height
    )
{
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    _internal_scale_e scale_type = SCALE;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    size_t i = 0;
    vsi_bool is_half_scale = FALSE;
    vsi_bool enable_copy = vsi_nn_kernel_param_get_int32( params, "enable_copy" );
    vsi_bool is_rgb888_sep = (vsi_bool)(inputs[1] != NULL);

    is_half_scale = (width == (int32_t)outputs[0]->attr.size[0] * 2) &&
                    (height == (int32_t)outputs[0]->attr.size[1] * 2);
    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (enable_copy)
    {
        scale_type = COPY;
    }
    else
    {
        if (is_no_range_change && is_half_scale)
        {
            scale_type = HALF;
        }
        else
        {
            scale_type = SCALE;
        }
    }

    key = PRE_PROCESS_RGB888_PLANAR_NHWC_HASH_KEY( input0_dtype, output_dtype, is_rgb888_sep, scale_type);

    for ( i = 0; i < _cnt_of_array(_pre_process_rgb888_planar_nhwc_kernel_map); i ++ )
    {
        if ( _pre_process_rgb888_planar_nhwc_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_pre_process_rgb888_planar_nhwc_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",
            _pre_process_rgb888_planar_nhwc_kernel_map[i].function_name );

        if (is_rgb888_sep)
        {
            kernel->info.parameters = _pre_process_rgb888_planar_sep_kernel_param_def;
            kernel->info.numParams = _cnt_of_array( _pre_process_rgb888_planar_sep_kernel_param_def );
        }
        else
        {
            kernel->info.parameters = _pre_process_rgb888_planar_kernel_param_def;
            kernel->info.numParams = _cnt_of_array( _pre_process_rgb888_planar_kernel_param_def );
        }

        if (enable_copy)
        {
            kernel->info.initialize = _pre_process_rgb888_planar_copy_initializer;
        }
        else if (scale_type == HALF)
        {
            kernel->info.initialize = _resize_rgb888_planar_initializer;
        }
        else
        {
            kernel->info.initialize = _pre_process_rgb888_planar_initializer;
        }
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                _pre_process_rgb888_planar_nhwc_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _pre_process_rgb888_planar_nhwc_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t* node_params = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* reshape_tensor = NULL;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    size_t param_count = _PRE_PROCESS_RGB888_PLANAR_SEP_PARAM_NUM;
    int32_t width  = vsi_nn_kernel_param_get_int32( params, "width" );
    int32_t height = vsi_nn_kernel_param_get_int32( params, "height" );
    float r_mean = vsi_nn_kernel_param_get_float32( params, "r_mean" );
    float g_mean = vsi_nn_kernel_param_get_float32( params, "g_mean" );
    float b_mean = vsi_nn_kernel_param_get_float32( params, "b_mean" );
    float r_scale = vsi_nn_kernel_param_get_float32( params, "r_scale" );
    float g_scale = vsi_nn_kernel_param_get_float32( params, "g_scale" );
    float b_scale = vsi_nn_kernel_param_get_float32( params, "b_scale" );
    int32_t reverse  = vsi_nn_kernel_param_get_int32( params, "reverse" );
    vsi_bool is_no_range_change = FALSE;

    input_num = inputs[1] == NULL ? 1 : input_num;
    param_count = inputs[1] == NULL ? _PRE_PROCESS_RGB888_PLANAR_PARAM_NUM : param_count;

    memcpy(shape, outputs[0]->attr.size, outputs[0]->attr.dim_num * sizeof(shape[0]));
    shape[0] *= shape[1];
    shape[1] = shape[2];
    shape[2] = 1;
    reshape_tensor = vsi_nn_reshape_tensor( graph,
            outputs[0], shape, outputs[0]->attr.dim_num );

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensor->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if ( width == (int32_t)inputs[0]->attr.size[0] && height == (int32_t)inputs[0]->attr.size[1] &&
         outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8 &&
         outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC &&
         (float)outputs[0]->attr.dtype.zero_point == r_mean && r_mean == g_mean && r_mean == b_mean &&
         vsi_nn_abs(outputs[0]->attr.dtype.scale - r_scale) < 1e-8 &&
         vsi_nn_abs(outputs[0]->attr.dtype.scale - g_scale) < 1e-8 &&
         vsi_nn_abs(outputs[0]->attr.dtype.scale - b_scale) < 1e-8)
    {
        is_no_range_change = TRUE;
    }

    status = _query_kernel( inputs, outputs, kernel, params, is_no_range_change, width, height );
    if ( VSI_SUCCESS == status)
    {
        node_params = (vsi_nn_kernel_node_param_t *)malloc(sizeof(vsi_nn_kernel_node_param_t) * param_count);
        CHECK_PTR_FAIL_GOTO( node_params, "Create buffer fail.", final );
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = inputs[1] == NULL ? 2 : 4;
            uint32_t scalar_index = index;
            int32_t scale_x = vsi_nn_kernel_param_get_int32( params, "scale_x" );
            int32_t scale_y = vsi_nn_kernel_param_get_int32( params, "scale_y" );
            int32_t left    = vsi_nn_kernel_param_get_int32( params, "left" );
            int32_t top     = vsi_nn_kernel_param_get_int32( params, "top" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, param_count,
                    inputs, input_num, &reshape_tensor, output_num );

            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_scale );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, param_count );
            index = scalar_index;
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
            vsi_nn_kernel_scalar_release( &node_params[index++] );
        }
    }

final:
    vsi_nn_safe_free(node_params);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( pre_process_rgb888_planar_nhwc, _setup )


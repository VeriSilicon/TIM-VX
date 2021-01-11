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

#define HASH_L2NORMALIZESCALE_HASH_KEY(AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, _image_2d) \
    ((AXIS << 28) | (IN1_DTYPE << 20) | (IN0_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

 #define HASH_L2NORMALIZESCALE_KERNEL_SOURCE_NAME(AXIS) \
    "l2normalizescale_axis"#AXIS

#define HASH_L2NORMALIZESCALE_KERNELS_2D( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        { HASH_L2NORMALIZESCALE_HASH_KEY(AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("evis.l2normalizescale_axis"#AXIS"_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE"_2D"), \
        HASH_L2NORMALIZESCALE_KERNEL_SOURCE_NAME(AXIS) },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _l2normalizescale_kernel_map[] =
{
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, F16, F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, I8 , F16, I8  )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, I8 , F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, U8 , F16, U8  )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, U8 , F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, I16, F16, I16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 0, I16, F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, F16, F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, I8 , F16, I8  )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, I8 , F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, U8 , F16, U8  )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, U8 , F16, F16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, I16, F16, I16 )
    HASH_L2NORMALIZESCALE_KERNELS_2D( 1, I16, F16, F16 )
};


/*
 * Kernel params
 */
static vx_param_description_t _l2normalizescale_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _L2NORMALIZESCALE_PARAM_NUM  _cnt_of_array( _l2normalizescale_kernel_param_def )

#define SCALAR_INPUT_AXIS          (3)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_l2normalizescale_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    int32_t     axis                           = 0;
    vsi_nn_kernel_tensor_attr_t *input_attr    = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr   = NULL;
    vsi_int_array_t * output_shape             = NULL;
    vsi_nn_kernel_dtype_e input_dtype          = F16;
    vsi_nn_kernel_dtype_e output_dtype         = F16;
    int32_t   input_fl      = 0;
    int32_t   inputZP       = 0;
    float     inputScale    = 1.0f;
    int32_t   output_fl     = 0;
    int32_t   outputZP      = 0;
    float     outputScale   = 1.0f;
    float     r_inputScale  = 1.0f;

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    output_shape  = output_attr->shape;
    input_dtype   = input_attr->dtype;
    output_dtype  = output_attr->dtype;

    if ( VSI_NN_KERNEL_QUANT_DFP == input_attr->quant )
    {
        input_fl   = input_attr->dfp.fl;
        if (input_fl >= 0)
        {
            inputScale = 1.0f / (float) ((int64_t)1 << input_fl);
        }
        else
        {
            inputScale = (float) ((int64_t)1 << -input_fl);
        }
    }
    else if ( VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant )
    {
        inputZP     = input_attr->asymm.zero_point;
        inputScale  = input_attr->asymm.scale;
    }

    if ( VSI_NN_KERNEL_QUANT_DFP == output_attr->quant )
    {
        output_fl   = output_attr->dfp.fl;
        if (output_fl >= 0)
        {
            outputScale = (float) ((int64_t)1 << output_fl);
        }
        else
        {
            outputScale = 1.0f / (float) ((int64_t)1 << -output_fl);
        }
    }
    else if ( VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant )
    {
        outputZP     = output_attr->asymm.zero_point;
        outputScale  = 1.0f / output_attr->asymm.scale;
    }


    r_inputScale = 1.0f / inputScale;

    if (1 == axis)
    {
        gpu_param.global_offset[0] = 0;
        gpu_param.global_offset[1] = 0;
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = 1;
    }
    else if (0 == axis)
    {
        gpu_param.global_offset[0] = 0;
        gpu_param.global_offset[1] = 0;
        gpu_param.global_scale[0]  = 16;
        gpu_param.global_scale[1]  = 1;
        gpu_param.local_size[0]    = 16;
        gpu_param.local_size[1]    = 1;
        gpu_param.global_size[0]   = 16;
        gpu_param.global_size[1]   = output_shape->data[1];
    }
    else
    {
        status = VSI_FAILURE;
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    {
        gpu_dp_inst_t UniFp16MulLo_dp4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00010000, 0x00030002, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t UniFp16MulHi_dp4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x01010101, // BSelt
            0x00050004, 0x00070006, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniIntegerSquareLo_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00010000, 0x00030002, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniIntegerSquareHi_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x00000000, // BSelt
            0x00050004, 0x00070006, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataSquareAddU32Lo_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x00000000, // BSelt
            0x00010000, 0x00030002, // BBin
            0x00005400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataSquareAddU32Hi_4x4 = {{
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00150004, 0x00370026, // ABin
            0x00000000, // BSelt
            0x00050004, 0x00070006, // BBin
            0x00005400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniUInt8SquareLo_4x4 = {{
            0x69696969, // TCfg
            0x40404040, // ASelt
            0x01110000, 0x03330222, // ABin
            0x54545454, // BSelt
            0x00010000, 0x00030002, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniUInt8SquareHi_4x4 = {{
            0x69696969, // TCfg
            0x40404040, // ASelt
            0x05550444, 0x07770666, // ABin
            0x54545454, // BSelt
            0x00050004, 0x00070006, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSumSqrt_16x1 = {{
            0x55555555, // TCfg
            0x55550000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x55550000, // BSelt
            0x76543210, 0x76543210, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniSumAll_16x1 = {{
            0x55555555, // TCfg
            0x55550000, // ASelt
            0x76543210, 0x76543210, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16};

        if (1 == axis)
        {
            int32_t L2NorS_depth = output_shape->data[1];
            status = vsi_nn_kernel_gpu_add_param( node, "L2NorS_depth",  &L2NorS_depth);
            if(F16 == input_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "UniFp16MulLo_dp4x4", &UniFp16MulLo_dp4x4);
                status |= vsi_nn_kernel_gpu_add_param( node, "UniFp16MulHi_dp4x4", &UniFp16MulHi_dp4x4);
            }
            else if(I8 == input_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "r_inputScale", &r_inputScale);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniDataSquareAddU32Lo_4x4", &uniDataSquareAddU32Lo_4x4);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniDataSquareAddU32Hi_4x4", &uniDataSquareAddU32Hi_4x4);
            }
            else if(I16 == input_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "r_inputScale", &r_inputScale);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniIntegerSquareLo_4x4", &uniIntegerSquareLo_4x4);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniIntegerSquareHi_4x4", &uniIntegerSquareHi_4x4);
            }
            else if(U8 == input_dtype)
            {
                status |= vsi_nn_kernel_gpu_add_param( node, "r_inputScale", &r_inputScale);
                status |= vsi_nn_kernel_gpu_add_param( node, "inputZP", &inputZP);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniUInt8SquareLo_4x4", &uniUInt8SquareLo_4x4);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniUInt8SquareHi_4x4", &uniUInt8SquareHi_4x4);
            }
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if (0 == axis)
        {
            int32_t inputWidth, inputWidthCount, inputWidthRemain256;
            inputWidth          = output_shape->data[0];
            inputWidthRemain256 = output_shape->data[0] % 256;
            inputWidthCount     = output_shape->data[0] / 256;
            vsi_nn_kernel_gpu_add_param( node, "inputWidth", &inputWidth);
            vsi_nn_kernel_gpu_add_param( node, "inputWidthRemain256", &inputWidthRemain256);
            vsi_nn_kernel_gpu_add_param( node, "inputWidthCount", &inputWidthCount);
            vsi_nn_kernel_gpu_add_param( node, "uniSumSqrt_16x1", &uniSumSqrt_16x1);
            if (I16 == input_dtype || I8 == input_dtype)
            {
                status = vsi_nn_kernel_gpu_add_param( node, "r_inputScale", &r_inputScale);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            else if(U8 == input_dtype)
            {
                float zP2x = 2 * (float)inputZP;
                float zpSqrt16x =  16 * (float)inputZP * (float)inputZP;
                status = vsi_nn_kernel_gpu_add_param( node, "r_inputScale", &r_inputScale);
                status |= vsi_nn_kernel_gpu_add_param( node, "zP2x", &zP2x);
                status |= vsi_nn_kernel_gpu_add_param( node, "zpSqrt16x", &zpSqrt16x);
                status |= vsi_nn_kernel_gpu_add_param( node, "uniSumAll_16x1", &uniSumAll_16x1);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
        }
    }

   {
        float IntergerScale = inputScale;
        float output_ZP      = (float)outputZP;
        gpu_dp_inst_t uniExtact8Bin_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataSubZPtoFp32Part0_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataSubZPtoFp32Part1_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniFp16toFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniFp16toFp32Hi_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

       IntergerScale = IntergerScale * outputScale;

        status  = vsi_nn_kernel_gpu_add_param( node, "IntergerScale", &IntergerScale);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &output_ZP);
        status |= vsi_nn_kernel_gpu_add_param( node, "inputZP", &inputZP);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniDataSubZPtoFp32Part0_4x4", &uniDataSubZPtoFp32Part0_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniDataSubZPtoFp32Part1_4x4", &uniDataSubZPtoFp32Part1_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32_4x4", &uniFp16toFp32_4x4);
        if (0 == axis)
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniFp16toFp32Hi_4x4", &uniFp16toFp32Hi_4x4);
        }

        if(F16 == output_dtype)
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bin_2x8", &uniExtractHalf8_2x8);
        }
        else
        {
            status |= vsi_nn_kernel_gpu_add_param( node, "uniExtact8Bin_2x8", &uniExtact8Bin_2x8);
        }

        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );

    return status;

} /* _l2normalizescale_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _l2normalizescale_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _l2normalizescale_kernel_map );
    vx_param_description_t * param_def  = _l2normalizescale_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _l2normalizescale_kernel_param_def );
    vx_kernel_initialize_f  initializer = _l2normalizescale_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype  = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_L2NORMALIZESCALE_HASH_KEY(axis, in0_dtype, in1_dtype, out_dtype, image_2d);

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
    vsi_nn_kernel_node_param_t node_params[_L2NORMALIZESCALE_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    int32_t  axis = 0;

    axis = vsi_nn_kernel_param_get_int32(params, "axis");

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num )
     || axis > 2)
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, axis, image_2d );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U32 = 0;
            border.constant_value.S16 = 0;
            border.constant_value.U8 = 0;
            if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
            {
                border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
            }
            status  = vsi_nn_kernel_node_set_border( node, &border );
            VSI_ASSERT( status == VSI_SUCCESS );
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _L2NORMALIZESCALE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(
                    graph, I32, &axis );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _L2NORMALIZESCALE_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( l2normalizescale, _setup )


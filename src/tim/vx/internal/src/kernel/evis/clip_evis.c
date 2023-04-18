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
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS


#define _CLIP_KERNEL_SOURCE(_input_type)      "clip_"#_input_type

#define STR(a) #a
// Add kernel hashtable here
#define CLIP_HASH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (_image_2d))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { CLIP_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0 ), \
          CVIVANTE_NAMESPACE("evis.clip_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
          _CLIP_KERNEL_SOURCE(IN_DTYPE) }

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE ) \
        { CLIP_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1 ), \
          CVIVANTE_NAMESPACE("evis.clip_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
          _CLIP_KERNEL_SOURCE(IN_DTYPE) }



typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _clip_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP(F16,  F16),
    PACK_KERNEL_MAP(F16,  I16),
    PACK_KERNEL_MAP(F16,  I8),
    PACK_KERNEL_MAP(F16,  U8),
    PACK_KERNEL_MAP(I16,  F16),
    PACK_KERNEL_MAP(I8,   F16),
    PACK_KERNEL_MAP(U8,   F16),
    PACK_KERNEL_MAP(I16,  I16),
    PACK_KERNEL_MAP(I8,   I8),
    PACK_KERNEL_MAP(U8,   U8),
    PACK_KERNEL_MAP_2D(F16,  F16),
    PACK_KERNEL_MAP_2D(F16,  I16),
    PACK_KERNEL_MAP_2D(F16,  I8),
    PACK_KERNEL_MAP_2D(F16,  U8),
    PACK_KERNEL_MAP_2D(I16,  F16),
    PACK_KERNEL_MAP_2D(I8,   F16),
    PACK_KERNEL_MAP_2D(U8,   F16),
    PACK_KERNEL_MAP_2D(I16,  I16),
    PACK_KERNEL_MAP_2D(I8,   I8),
    PACK_KERNEL_MAP_2D(U8,   U8),
};


/*
 * Kernel params
 */
static vx_param_description_t _clip_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _CLIP_PARAM_NUM  _cnt_of_array( _clip_kernel_param_def )

#define SCALAR_MIN_VALUE          (2)
#define SCALAR_MAX_VALUE          (3)

#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_clip_initializer)
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
    vsi_size_array_t * out_shape                 = NULL;
    vsi_nn_kernel_dtype_e        input_dtype    = F16;
    vsi_nn_kernel_dtype_e        output_dtype   = F16;
    float     minVal          = 1.0f;
    float     maxVal          = 1.0f;
    float     scaleIn         = 1.0f;
    float     scaleOut        = 1.0f;
    int32_t   output_ZP       = 0;
    int32_t   input_ZP        = 0;
    int32_t   srcFixPointPos  = 0;
    int32_t   dstFixPointPos  = 0;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );
    out_shape  = output_attr->shape;
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_MIN_VALUE], &(minVal));
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_MAX_VALUE], &(maxVal));
    input_dtype  = input_attr->dtype;
    output_dtype = output_attr->dtype;

    if ((F16 == input_dtype)
        || (I16 == input_dtype)
        || (BF16 == input_dtype)
       )
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }
    else
    {
        gpu_param.global_scale[0]  = 16;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }

    gpu_param.dim = out_shape->size < 3 ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;


    if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        srcFixPointPos   = input_attr->dfp.fl;
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant)
    {
        input_ZP         = input_attr->asymm.zero_point;
        scaleIn          = input_attr->asymm.scale;
    }

    if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        dstFixPointPos   = output_attr->dfp.fl;
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant)
    {
        output_ZP        = output_attr->asymm.zero_point;
        scaleOut         = output_attr->asymm.scale;
    }

    if ((F16 == input_dtype &&
        (F16 == output_dtype || I8 == output_dtype
       || I16 == output_dtype || U8 == output_dtype))
       || (BF16 == input_dtype && BF16 == output_dtype)
       )
    {
        uint16_t minTmp   = 0;
        uint16_t maxTmp   = 0;
        uint32_t packedMin = 0;
        uint32_t packedMax = 0;
        uint32_t packedMinData_FP16[4];
        uint32_t packedMaxData_FP16[4];
        uint32_t i;

        if (BF16 == input_dtype)
        {
            minTmp = vsi_nn_Fp32ToBFp16(minVal);
            maxTmp = vsi_nn_Fp32ToBFp16(maxVal);
        }
        else
        {
            minTmp = vsi_nn_Fp32toFp16(minVal);
            maxTmp = vsi_nn_Fp32toFp16(maxVal);
        }

        packedMin = (minTmp << 16) | (minTmp);
        packedMax = (maxTmp << 16) | (maxTmp);

        for (i = 0;i < 4; i++)
        {
            packedMinData_FP16[i] = packedMin;
            packedMaxData_FP16[i] = packedMax;
        }

        status  = vsi_nn_kernel_gpu_add_param( node, "packedMinData_FP16", packedMinData_FP16);
        status |= vsi_nn_kernel_gpu_add_param( node, "packedMaxData_FP16", packedMaxData_FP16);
        if (I8 == output_dtype ||  I16 == output_dtype)
        {
            gpu_dp_inst_t uniConvertF16toInt_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16};
            if (dstFixPointPos <= 0)
            {
                uniConvertF16toInt_2x8.data[7] |= vsi_nn_min((-dstFixPointPos) & 0x1F, MAX_POST_SHIFT_BITS);
            }
            else
            {
                uint32_t lo_part    = vsi_nn_min(((int64_t)1 << dstFixPointPos), MAX_MULTIPLIER_NUM);
                uint32_t multiplier = lo_part;
                uint32_t j          = 0;

                for (j = 0; j < 8; j++)
                {
                    uniConvertF16toInt_2x8.data[j + 8] = multiplier;
                }
            }
            status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertF16toInt_2x8", &uniConvertF16toInt_2x8);
        }
        else if (U8 == output_dtype)
        {
            uint32_t  multAndoutZP[2]    = {0};
            uint16_t  M0                 = 0;
            int32_t   postShift          = 0;
            gpu_dp_inst_t  uniDataMulAndPostShift_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111119, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_quantize_multiplier_16bit(scaleIn / scaleOut, &M0, &postShift);

            multAndoutZP[0] = (uint32_t)(M0);
            multAndoutZP[1] = (uint32_t)(output_ZP << postShift );

            uniDataMulAndPostShift_2x8.data[7] |= (postShift & 0x1F);
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP", multAndoutZP);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniDataMulAndPostShift_2x8", &uniDataMulAndPostShift_2x8);
        }
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (I8 == input_dtype
         && (I8 == output_dtype || F16 == output_dtype))
    {
        int32_t packedMin = 0;
        int32_t packedMax = 0;
        int32_t packedMinData[4];
        int32_t packedMaxData[4];
        gpu_dp_inst_t uniConvertIntegerLo_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertIntegerHi_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        if (srcFixPointPos > dstFixPointPos)
        {
            int32_t  postshift      = vsi_nn_min(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertIntegerLo_2x8.data[7] |= (postshift & 0x1F);
            uniConvertIntegerHi_2x8.data[7] |= (postshift & 0x1F);
        }
        else
        {
            uint32_t multiplier = vsi_nn_min((int64_t)1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            uint32_t i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertIntegerLo_2x8.data[i + 8] = multiplier;
                uniConvertIntegerHi_2x8.data[i + 8] = multiplier;
            }
        }

        if (F16 == output_dtype)
        {
            uint16_t minData   = 0;
            uint16_t maxData   = 0;
            minData = vsi_nn_Fp32toFp16(minVal);
            maxData = vsi_nn_Fp32toFp16(maxVal);
            packedMin = (minData << 16) | (minData);
            packedMax = (maxData << 16) | (maxData);
        }
        else
        {
            uint8_t minData   = 0;
            uint8_t maxData   = 0;
            minData =  (uint8_t)vsi_nn_Fp32ToDFP(minVal, (int8_t)dstFixPointPos, VSI_NN_TYPE_INT8);
            maxData =  (uint8_t)vsi_nn_Fp32ToDFP(maxVal, (int8_t)dstFixPointPos, VSI_NN_TYPE_INT8);
            packedMin = (minData << 24) | (minData << 16) | (minData << 8) | (minData);
            packedMax = (maxData << 24) | (maxData << 16) | (maxData << 8) | (maxData);
        }

        packedMinData[0] = packedMinData[1] = packedMinData[2] = packedMinData[3] = packedMin;
        packedMaxData[0] = packedMaxData[1] = packedMaxData[2] = packedMaxData[3] = packedMax;

        status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertIntegerLo_2x8", &uniConvertIntegerLo_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertIntegerHi_2x8", &uniConvertIntegerHi_2x8);

        status |= vsi_nn_kernel_gpu_add_param( node, "packedMinData", packedMinData);
        status |= vsi_nn_kernel_gpu_add_param( node, "packedMaxData", packedMaxData);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (I16 == input_dtype
         && (I16 == output_dtype || F16 == output_dtype))
    {
        uint16_t minData  = 0;
        uint16_t maxData  = 0;
        int32_t packedMin = (minData << 16) | (minData);
        int32_t packedMax = (maxData << 16) | (maxData);
        int32_t packedMinData[4];
        int32_t packedMaxData[4];
        gpu_dp_inst_t uniConvertIntegerLo_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        if (F16 == output_dtype)
        {
            minData = vsi_nn_Fp32toFp16(minVal);
            maxData = vsi_nn_Fp32toFp16(maxVal);
        }
        else
        {
            minData =  (uint16_t)vsi_nn_Fp32ToDFP(minVal, (int8_t)dstFixPointPos, VSI_NN_TYPE_INT16);
            maxData =  (uint16_t)vsi_nn_Fp32ToDFP(maxVal, (int8_t)dstFixPointPos, VSI_NN_TYPE_INT16);
        }

        packedMin = (minData << 16) | (minData);
        packedMax = (maxData << 16) | (maxData);

        packedMinData[0] = packedMinData[1] = packedMinData[2] = packedMinData[3] = packedMin;
        packedMaxData[0] = packedMaxData[1] = packedMaxData[2] = packedMaxData[3] = packedMax;

        if (srcFixPointPos > dstFixPointPos)
        {
            int32_t  postshift      = vsi_nn_min(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertIntegerLo_2x8.data[7] |= (postshift & 0x1F);
        }
        else
        {
            uint32_t multiplier = vsi_nn_min((int64_t)1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            uint32_t i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertIntegerLo_2x8.data[i + 8] = multiplier;
            }
        }

        status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertIntegerLo_2x8", &uniConvertIntegerLo_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "packedMinData", packedMinData);
        status |= vsi_nn_kernel_gpu_add_param( node, "packedMaxData", packedMaxData);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (U8 == input_dtype
         && (U8 == output_dtype || F16 == output_dtype))
    {
        int32_t   packedMin        = 0;
        int32_t   packedMax        = 0;
        int32_t   packedMinData[4];
        int32_t   packedMaxData[4];
        float     uint8Scale = scaleIn / scaleOut;
        uint16_t  M0                   = 0;
        int32_t   postShift            = 0;
        uint32_t  multAndoutZP[2]      = {0};
        gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8MulAndPostShift_Hi_2x8 = {{
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x1b1a1918, 0x1f1e1d1c, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_quantize_multiplier_16bit(uint8Scale, &M0, &postShift);
        multAndoutZP[0] = (uint32_t)(M0);
        multAndoutZP[1] = (uint32_t)((output_ZP << postShift) - input_ZP * M0);

        uniU8MulAndPostShift_Lo_2x8.data[7] |= (postShift & 0x1F);
        uniU8MulAndPostShift_Hi_2x8.data[7] |= (postShift & 0x1F);

        if (F16 == output_dtype)
        {
            uint16_t   minData          = 0;
            uint16_t   maxData          = 0;
            minData = vsi_nn_Fp32toFp16(minVal);
            maxData = vsi_nn_Fp32toFp16(maxVal);
            packedMin = (minData << 16) | (minData);
            packedMax = (maxData << 16) | (maxData);
        }
        else
        {
            uint8_t   minData          = 0;
            uint8_t   maxData          = 0;
            minData = (uint8_t)vsi_nn_Fp32ToAffine(minVal, scaleOut, output_ZP, VSI_NN_TYPE_UINT8);
            maxData = (uint8_t)vsi_nn_Fp32ToAffine(maxVal, scaleOut, output_ZP, VSI_NN_TYPE_UINT8);
            packedMin = (minData << 24) | (minData << 16) | (minData << 8) | (minData);
            packedMax = (maxData << 24) | (maxData << 16) | (maxData << 8) | (maxData);
        }

        packedMinData[0] = packedMinData[1] = packedMinData[2] = packedMinData[3] = packedMin;
        packedMaxData[0] = packedMaxData[1] = packedMaxData[2] = packedMaxData[3] = packedMax;

        status  = vsi_nn_kernel_gpu_add_param( node, "uniU8MulAndPostShift_Lo_2x8", &uniU8MulAndPostShift_Lo_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniU8MulAndPostShift_Hi_2x8", &uniU8MulAndPostShift_Hi_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP", multAndoutZP);

        status |= vsi_nn_kernel_gpu_add_param( node, "packedMinData", packedMinData);
        status |= vsi_nn_kernel_gpu_add_param( node, "packedMaxData", packedMaxData);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    SAFE_FREE_TENSOR_ATTR(input_attr);
    return status;
} /* _clip_initializer() */

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
    const _kernel_map_type * kernel_map = _clip_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _clip_kernel_map );
    vx_param_description_t * param_def  = _clip_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _clip_kernel_param_def );
    vx_kernel_initialize_f  initializer = _clip_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = CLIP_HASH_KEY( in_dtype, out_dtype, image_2d );

    if ( ( in_dtype == I8 || in_dtype == I16 ) &&
         ( inputs[0]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_DFP &&
           inputs[0]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_NONE ) )
    {
        return VSI_FAILURE;
    }

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
    vsi_nn_kernel_node_param_t node_params[_CLIP_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    float   min_value  = vsi_nn_kernel_param_get_float32( params, "min_value" );
    float   max_value  = vsi_nn_kernel_param_get_float32( params, "max_value" );
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t new_rank = 0;
    vsi_bool ret = TRUE;

    ret = vsi_nn_kernel_optimize_element_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num, shape, &new_rank);

    if ( ret )
    {
        return NULL;
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], shape, new_rank );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], shape, new_rank );

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[0]->attr.size,
                reshape_tensors[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (reshape_tensors[0]->attr.dim_num == 2 || reshape_tensors[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, reshape_tensors, &reshape_tensors[1], image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CLIP_PARAM_NUM,
                    reshape_tensors, input_num, &reshape_tensors[1], output_num );
            node_params[SCALAR_MIN_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &min_value );
            node_params[SCALAR_MAX_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &max_value );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CLIP_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MIN_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MAX_VALUE] );
        }
    }

    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( clip, _setup )

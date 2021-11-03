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
#pragma once
#ifndef _OPENVX_EXT_LIBNNEXT_H_
#define _OPENVX_EXT_LIBNNEXT_H_
#include <VX/vx.h>
#include <VX/vx_types.h>

#define gcoMATH_MIN(X, Y) (((X) < (Y))?(X):(Y))
#define gcoMATH_MAX(X, Y) (((X) > (Y))?(X):(Y))
#define DIM_SIZE 4

#ifdef __cplusplus
extern "C" {
#endif

#define VIVANTE_NAMESPACE               "com.vivantecorp.extension"
#define CVIVANTE_NAMESPACE(str)         (VIVANTE_NAMESPACE "." str)

/**
 *  Assigned from Khronos, vendors control their own
 */
#define VX_LIBRARY_LIBNNEXT             (0x3)
#define KERNEL_ID_OVXLIB_START          (VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + 0x1)
#define KERNEL_ID_OVXLIB_RESERVED       (KERNEL_ID_OVXLIB_START + 0xFFF)
/**
 * Use kernel id placeholder to tell ovxlib
 * generate a unique id for this kernel.
 */
#define KERNEL_ID_PLACEHOLDER           (0x1000)

/*! if there are more than 1 kernel in solution
the KERNEL_ENUM_LIBNNEXT_OFFSET must be modified keep different for any kernel
*/
enum vx_kernel_libnnext_offset_e
{
    KERNEL_ENUM_LIBNNEXT_OFFSET = 1,
    KERNEL_ENUM_PREMUTE_OFFSET,
    KERNEL_ENUM_PRIORBOX_OFFSET = 2 + KERNEL_ENUM_PREMUTE_OFFSET,
    KERNEL_ENUM_FLATTEN_OFFSET,
    KERNEL_ENUM_L2NORMALIZESCALE_OFFSET,
    KERNEL_ENUM_PARAMETRICRELU_OFFSET,
    KERNEL_ENUM_PREBBOX_OFFSET = 3 + KERNEL_ENUM_PARAMETRICRELU_OFFSET,
    KERNEL_ENUM_ADD_RELU_KERNEL_OFFSET,
    KERNEL_ENUM_POOLING_WITH_ARGMAX_OFFSET,
    KERNEL_ENUM_UNPOOLING_OFFSET = 2 + KERNEL_ENUM_POOLING_WITH_ARGMAX_OFFSET,
    KERNEL_ENUM_ARGMAX_OFFSET = 2 + KERNEL_ENUM_UNPOOLING_OFFSET,
    KERNEL_ENUM_ALEXNET_GEMM_OFFSET = 2 + KERNEL_ENUM_ARGMAX_OFFSET,
    KERNEL_ENUM_IMG2COL_DILATED_OFFSET,
    KERNEL_ENUM_IMG2COL_DILATED_INT8_OFFSET,
    KERNEL_ENUM_ALEXNET_GEMM_INT8_OFFSET,
    KERNEL_ENUM_ELTWISE_MAX,
    KERNEL_ENUM_FULLYCONNECTED_AXIS2,
    KERNEL_ENUM_TENSORCROP_INT16,
    KERNEL_ENUM_TENSORCROP_INT8,
    KERNEL_ENUM_TENSORCROP_INT16_FP16,
    KERNEL_ENUM_DROPOUT,
    KERNEL_ENUM_SHUFFLECHANNEL,
    KERNEL_ENUM_RESIZE,
    KERNEL_ENUM_REVERSE,
    KERNEL_ENUM_RESIZE_16BITS_DOWNSAMPLE_QUARTER,
    KERNEL_ENUM_RESIZE_8BITS_DOWNSAMPLE_QUARTER,
    KERNEL_ENUM_SCALE,
    KERNEL_ENUM_TENSORREVERSE,
    KERNEL_ENUM_TENSORELU_OFFSET,
    KERNEL_ENUM_SPACE2BATCH,
    KERNEL_ENUM_BATCH2SPACE,
    KERNEL_ENUM_SPACE2DEPTH,
    KERNEL_ENUM_IMAGEPROCESS,
    KERNEL_ENUM_SCALETOTENSOR,
    KERNEL_ENUM_GEMM,
    KERNEL_ENUM_LAYERNORM,
    KERNEL_ENUM_LAYERNORMFP16TOU8_OFFSET,
    KERNEL_ENUM_REDUCE,
    KERNEL_ENUM_INSTANCENORM,
    KERNEL_ENUM_TENSORSTACKCONCAT,
    KERNEL_ENUM_TENSORSTACKCONCAT8BITS_OFFSET,
    KERNEL_ENUM_SIGNALFRAME,
    KERNEL_ENUM_RELATIONALOPS,
    KERNEL_ENUM_SYNC_HOST,
    KERNEL_ENUM_POW,
    KERNEL_ENUM_FLOORDIV,
    KERNEL_ENUM_SPATIAL_TRANSFORMER,
    KERNEL_ENUM_LOGICAL_OPS,
    KERNEL_ENUM_SELECT,
    KERNEL_ENUM_LSTMUNIT_ACTIVATION,
    KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM,
    KERNEL_ENUM_STACK,
    KERNEL_ENUM_GRAYSCALETOTENSOR,
    KERNEL_ENUM_NEG,
    KERNEL_ENUM_EXP,
    KERNEL_ENUM_CLIP,
    KERNEL_ENUM_PRE_PROCESS_GRAY,
    KERNEL_ENUM_UNSTACK,
    KERNEL_ENUM_PRE_PROCESS_RGB,
    KERNEL_ENUM_ADDN,
    KERNEL_ENUM_PRE_PROCESS_YUV420,
    KERNEL_ENUM_CONV2D,
    KERNEL_ENUM_EXTRA_ENDING,
    KERNEL_ENUM_GATHER,
    KERNEL_ENUM_TILE,
    KERNEL_ENUM_TOPK,
    KERNEL_ENUM_PRE_PROCESS_BGRA,
    KERNEL_ENUM_LOGICAL_NOT,
    KERNEL_ENUM_SIN,
    KERNEL_ENUM_LOG,
    KERNEL_ENUM_ARGMIN,
    KERNEL_ENUM_ROI_ALIGN,
    KERNEL_ENUM_HEATMAP_MAX_KEYPOINT,
    KERNEL_ENUM_AXIS_ALIGNED_BBOX_TRANSFORM,
    KERNEL_ENUM_BOX_WITH_NMS_LIMIT,
    KERNEL_ENUM_GENERATE_PROPOSALS,
    KERNEL_ENUM_DETECTION_POSTPROCESS,
    KERNEL_ENUM_RANDOM_MULTINOMIAL,
    KERNEL_ENUM_LOG_SOFTMAX,
    KERNEL_ENUM_RELU_KERAS_INTERNAL,
    KERNEL_ENUM_DECONV2D,
    KERNEL_ENUM_REDUCEMAX_INTERNAL,
    KERNEL_ENUM_REDUCEMIN_INTERNAL,
    KERNEL_ENUM_REDUCEPROD_INTERNAL,
    KERNEL_ENUM_REDUCEALL_INTERNAL,
    KERNEL_ENUM_REDUCEANY_INTERNAL,
    KERNEL_ENUM_RESIZE_INTERNAL,
    KERNEL_ENUM_RESIZE_NEAREST_INTERNAL,
    KERNEL_ENUM_PRE_PROCESS_YUV444,
};

//! [KERNEL NAME]
#define VX_KERNEL_NAME_PERMUTECWH                          VIVANTE_NAMESPACE ".vxcPermuteCWH"
#define VX_KERNEL_NAME_PERMUTECHW                          VIVANTE_NAMESPACE ".vxcPermuteCWH"
#define VX_KERNEL_NAME_PRIORBOX                            VIVANTE_NAMESPACE ".vxcPriorBox"
#define VX_KERNEL_NAME_FLATTEN                             VIVANTE_NAMESPACE ".flatten"
//! l2normalizscale kernel
#define VX_KERNEL_NAME_L2NORMALIZESCALE                    VIVANTE_NAMESPACE ".vxcL2NormalizeScale"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI1_F16_2D    \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis1_F16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI1_I8_2D     \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis1_I8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI1_U8_2D     \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis1_U8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI1_I16_2D    \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis1_I16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_F16TOF16_2D        \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_F16toF16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_I8TOI8_2D          \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_I8toI8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_I8TOF16_2D         \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_I8toF16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_U8TOU8_2D          \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_U8toU8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_U8TOF16_2D         \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_U8toF16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_I16TOI16_2D        \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_I16toI16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI1_I16TOF16_2D        \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis1_I16toF16_2D"

#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI0_F16_2D    \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis0_F16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI0_I8_2D     \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis0_I8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI0_U8_2D     \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis0_U8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_AXI0_I16_2D    \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_SumRsqrt_axis0_I16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_F16TOF16_2D        \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_F16toF16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_I8TOI8_2D          \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_I8toI8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_I8TOF16_2D         \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_I8toF16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_U8TOU8_2D          \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_U8toU8_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_U8TOF16_2D         \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_U8toF16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_I16TOI16_2D        \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_I16toI16_2D"
#define VX_KERNEL_NAME_L2NORMSCALE_AXI0_I16TOF16_2D        \
                                    VIVANTE_NAMESPACE ".vxcL2NormScale_axis0_I16toF16_2D"
//! Prelu Kernel
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I8F16TOI8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I8F16toI8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I8F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I8F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I16F16TOI16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I16F16toI16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I16F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I16F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_U8F16TOU8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_U8F16toU8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_U8F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_U8F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOU8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toU8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOI8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toI8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOI16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toI16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_BF16F16TOBF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_BF16F16toBF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_BF16BF16TOBF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_BF16BF16toBF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I8F16TOI8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I8F16toI8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I8F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I8F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I16F16TOI16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I16F16toI16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_I16F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_I16F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_U8F16TOU8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_U8F16toU8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_U8F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_U8F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOU8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toU8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOI8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toI8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_F16F16TOI16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_F16F16toI16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_BF16F16TOBF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_BF16F16toBF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI0_BF16BF16TOBF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis0_BF16BF16toBF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I8F16TOI8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I8F16toI8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I8F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I8F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I16F16TOI16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I16F16toI16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I16F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I16F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_U8F16TOU8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_U8F16toU8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_U8F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_U8F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOU8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toU8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOI8 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toI8"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOI16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toI16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_BF16F16TOBF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_BF16F16toBF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_BF16BF16TOBF16 \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_BF16BF16toBF16"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I8F16TOI8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I8F16toI8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I8F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I8F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I16F16TOI16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I16F16toI16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_I16F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_I16F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_U8F16TOU8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_U8F16toU8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_U8F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_U8F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOU8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toU8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOI8_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toI8_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_F16F16TOI16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_F16F16toI16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_BF16F16TOBF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_BF16F16toBF16_2D"
#define VX_KERNEL_NAME_PARAMETRICRELU_AXI1_BF16BF16TOBF16_2D \
                                    VIVANTE_NAMESPACE ".vxcParametricRelu_axis1_BF16BF16toBF16_2D"

#define VX_KERNEL_NAME_PREBBOX                             VIVANTE_NAMESPACE ".preBBoxVXC"
#define VX_KERNEL_NAME_ADD_RELU_KERNEL                     VIVANTE_NAMESPACE ".addRelu"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX                 VIVANTE_NAMESPACE ".poolingWithArgmax"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8            VIVANTE_NAMESPACE ".poolingWithArgmaxInt8"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8_OPT        VIVANTE_NAMESPACE ".poolingWithArgmaxInt8_Int8_opt"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8_INT8       VIVANTE_NAMESPACE ".poolingWithArgmaxInt8_Int8"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16           VIVANTE_NAMESPACE ".poolingWithArgmaxInt16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_INT16     \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxInt16_int16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_OPT       \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxInt16_s2k2p0_opt"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_FP16      \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxInt16_fp16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_AXINT16   \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxInt16_axI16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8           VIVANTE_NAMESPACE ".poolingWithArgmaxUint8_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8_FP16      \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxUint8_fp16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8_FP16_FP16 \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxUint8_fp16_fp16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8_FP16       \
                                VIVANTE_NAMESPACE ".poolingWithArgmaxInt8_fp16_s2k2p0"
#define VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8_2D        VIVANTE_NAMESPACE ".poolingWithArgmaxU8_s2k2p0_2D"
#define VX_KERNEL_NAME_UNPOOLING                           VIVANTE_NAMESPACE ".unpooling"
#define VX_KERNEL_NAME_UNPOOLING_INT8                      VIVANTE_NAMESPACE ".unpoolingInt8"
#define VX_KERNEL_NAME_UNPOOLING_INT8_INT8                 VIVANTE_NAMESPACE ".unpoolingInt8_Int8"
#define VX_KERNEL_NAME_UNPOOLING_INT8_INT8_OPT             VIVANTE_NAMESPACE ".unpoolingInt8_Int8_opt"
#define VX_KERNEL_NAME_UNPOOLING_UINT8                     VIVANTE_NAMESPACE ".unpoolingUint8_Uint8"
#define VX_KERNEL_NAME_UNPOOLING_INT16_INT16               VIVANTE_NAMESPACE ".unpoolingInt16_Int16"
#define VX_KERNEL_NAME_UNPOOLING_INT16_INT16_OPT           VIVANTE_NAMESPACE ".unpoolingInt16_Int16_opt"
#define VX_KERNEL_NAME_UNPOOLING_INT16_INT16_AXINT16       VIVANTE_NAMESPACE ".unpoolingInt16_Int16_axI16"
#define VX_KERNEL_NAME_UNPOOLING_INT16_FP16_AXINT16        VIVANTE_NAMESPACE ".unpoolingInt16_Fp16_axI16"
#define VX_KERNEL_NAME_UNPOOLING_INT16_FP16                VIVANTE_NAMESPACE ".unpoolingInt16_Fp16"
#define VX_KERNEL_NAME_UNPOOLING_FP16_UINT8                VIVANTE_NAMESPACE ".unpoolingFp16_Uint8"
#define VX_KERNEL_NAME_UNPOOLING_FP16_INT8                 VIVANTE_NAMESPACE ".unpoolingFp16_Int8"
#define VX_KERNEL_NAME_UNPOOLING_FP16_INT16                VIVANTE_NAMESPACE ".unpoolingFp16_Int16"
#define VX_KERNEL_NAME_UNPOOLING_FP16FP16_UINT8            VIVANTE_NAMESPACE ".unpoolingFp16Fp16_Uint8"
#define VX_KERNEL_NAME_UNPOOLING_UINT8_FP16                VIVANTE_NAMESPACE ".unpoolingUint8_Fp16"
#define VX_KERNEL_NAME_UNPOOLING_INT8_FP16                 VIVANTE_NAMESPACE ".unpoolingInt8_Fp16"
#define VX_KERNEL_NAME_UNPOOLING_UINT8_2D                  VIVANTE_NAMESPACE ".unpoolingUint8_Uint8_2D"
#define VX_KERNEL_NAME_UNPOOLING_FP16_UINT8_2D             VIVANTE_NAMESPACE ".unpoolingFp16_Uint8_2D"
#define VX_KERNEL_NAME_ALEXNET_GEMM                        VIVANTE_NAMESPACE ".alexNet_gemmVXC"
#define VX_KERNEL_NAME_IMG2COL_DILATED                     VIVANTE_NAMESPACE ".img2col_dilatedVXC"
#define VX_KERNEL_NAME_IMG2COL_DILATED_INT8                VIVANTE_NAMESPACE ".img2col_dilated_int8VXC"
#define VX_KERNEL_NAME_ALEXNET_GEMM_INT8                   VIVANTE_NAMESPACE ".alexNet_gemm_int8VXC"
#define VX_KERNEL_NAME_FULLYCONNECTED_AXIS2                VIVANTE_NAMESPACE ".vxcFullyConnected_Axis2"
#define VX_KERNEL_NAME_TENSORCROP_INT16                    VIVANTE_NAMESPACE ".vxcTensorCrop_Int16"
#define VX_KERNEL_NAME_TENSORCROP_INT8                     VIVANTE_NAMESPACE ".vxcTensorCrop_Int8"
#define VX_KERNEL_NAME_TENSORCROP_INT16_FP16               VIVANTE_NAMESPACE ".vxcTensorCrop_Int16_Fp16"
#define VX_KERNEL_NAME_SHUFFLECHANNEL                      VIVANTE_NAMESPACE ".shuffleChannelVXC"
#define VX_KERNEL_NAME_SHUFFLECHANNEL8BITS                 VIVANTE_NAMESPACE ".shuffleChannel8BitsVXC"
#define VX_KERNEL_NAME_SHUFFLECHANNEL16BITS_AXIS1          VIVANTE_NAMESPACE ".shuffleChannel16Bits_Axis1"
#define VX_KERNEL_NAME_SHUFFLECHANNEL8BITS_AXIS1           VIVANTE_NAMESPACE ".shuffleChannel8Bits_Axis1"
#define VX_KERNEL_NAME_RESIZE_16BITS_DOWNSAMPLE_QUARTER    \
                                VIVANTE_NAMESPACE ".resize_16bits_downsample_quarter"
#define VX_KERNEL_NAME_RESIZE_8BITS_DOWNSAMPLE_QUARTER     \
                                VIVANTE_NAMESPACE ".resize_8bits_downsample_quarter"
#define VX_KERNEL_NAME_SCALE_FP16                          VIVANTE_NAMESPACE ".scale_fp16"
#define VX_KERNEL_NAME_TENSORREVERSE                       VIVANTE_NAMESPACE ".tensorReverse_axis0_fp16"
#define VX_KERNEL_NAME_SPACE2DEPTH_INT16_INT16             VIVANTE_NAMESPACE ".vxcReorg2_fp16_fp16_sx2_sy1"
#define VX_KERNEL_NAME_SCALETOTENSOR_FP16                  VIVANTE_NAMESPACE ".ScaletoTensor_Fp16"
#define VX_KERNEL_NAME_SCALETOTENSOR_INT8                  VIVANTE_NAMESPACE ".ScaletoTensor_Int8"
#define VX_KERNEL_NAME_SCALETOTENSOR_FP16_COPY             VIVANTE_NAMESPACE ".ScaletoTensor_Fp16_copy"
#define VX_KERNEL_NAME_SCALETOTENSOR_INT8_COPY             VIVANTE_NAMESPACE ".ScaletoTensor_Int8_copy"
#define VX_KERNEL_NAME_SCALETOTENSOR_INT16                 VIVANTE_NAMESPACE ".ScaletoTensor_Int16"
#define VX_KERNEL_NAME_SCALETOTENSOR_INT16_COPY            VIVANTE_NAMESPACE ".ScaletoTensor_Int16_copy"
#define VX_KERNEL_NAME_SCALETOTENSOR_UINT8                 VIVANTE_NAMESPACE ".ScaletoTensor_UInt8"
#define VX_KERNEL_NAME_SCALETOTENSOR_UINT8_COPY            VIVANTE_NAMESPACE ".ScaletoTensor_UInt8_copy"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_FP16              VIVANTE_NAMESPACE ".GrayScaletoTensor_Fp16"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT8              VIVANTE_NAMESPACE ".GrayScaletoTensor_Int8"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_FP16_COPY         VIVANTE_NAMESPACE ".GrayScaletoTensor_Fp16_copy"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT8_COPY         VIVANTE_NAMESPACE ".GrayScaletoTensor_Int8_copy"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT16             VIVANTE_NAMESPACE ".GrayScaletoTensor_Int16"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT16_COPY        VIVANTE_NAMESPACE ".GrayScaletoTensor_Int16_copy"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_UINT8             VIVANTE_NAMESPACE ".GrayScaletoTensor_UInt8"
#define VX_KERNEL_NAME_GRAYSCALETOTENSOR_UINT8_COPY        VIVANTE_NAMESPACE ".GrayScaletoTensor_UInt8_copy"
#define VX_KERNEL_NAME_TENSORSTACKCONCAT                   VIVANTE_NAMESPACE ".vxcTensorStackConcat"
#define VX_KERNEL_NAME_TENSORSTACKCONCAT8BITS              VIVANTE_NAMESPACE ".vxcTensorStackConcat8Bits"
#define VX_KERNEL_NAME_SIGNALFRAME_WIDTH                   VIVANTE_NAMESPACE ".vxcSignalFrame_width"
#define VX_KERNEL_NAME_SIGNALFRAME_HEIGHT                  VIVANTE_NAMESPACE ".vxcSignalFrame_height"
#define VX_KERNEL_NAME_SIGNALFRAME_CHANNEL                 VIVANTE_NAMESPACE ".vxcSignalFrame_channel"
#define VX_KERNEL_NAME_SIGNALFRAME_WIDTH_8BITS             VIVANTE_NAMESPACE ".vxcSignalFrame_width_8bit"
#define VX_KERNEL_NAME_SIGNALFRAME_HEIGHT_8BITS            VIVANTE_NAMESPACE ".vxcSignalFrame_height_8bit"
#define VX_KERNEL_NAME_SIGNALFRAME_CHANNEL_8BITS           VIVANTE_NAMESPACE ".vxcSignalFrame_channel_8bit"
#define VX_KERNEL_NAME_FLOORDIV_FP16                       VIVANTE_NAMESPACE ".vxcTensorFloorDiv_Fp16"
#define VX_KERNEL_NAME_FLOORDIV_INT16                      VIVANTE_NAMESPACE ".vxcTensorFloorDiv_Int16"
#define VX_KERNEL_NAME_FLOORDIV_INT8                       VIVANTE_NAMESPACE ".vxcTensorFloorDiv_Int8"
#define VX_KERNEL_NAME_FLOORDIV_UINT8                      VIVANTE_NAMESPACE ".vxcTensorFloorDiv_Uint8"
#define VX_KERNEL_NAME_SPATIAL_TRANSFORMER                 VIVANTE_NAMESPACE ".vxcTransform_Gemm_F16toF16"
#define VX_KERNEL_NAME_TRANSFORM_SETUP_THRES_F16TOF16      VIVANTE_NAMESPACE ".vxcTransform_setupThres_F16toF16"
#define VX_KERNEL_NAME_TRANSFORM_INTERP_F16TOF16_2D        VIVANTE_NAMESPACE ".vxcTransform_InterP_F16toF16_2D"
#define VX_KERNEL_NAME_TRANSFORM_INTERP_F16TOF16           VIVANTE_NAMESPACE ".vxcTransform_InterP_F16toF16"
#define VX_KERNEL_NAME_LOGICAL_OR_INT16                    VIVANTE_NAMESPACE ".vxcTensorLogical_or_int16"
#define VX_KERNEL_NAME_LOGICAL_OR_INT8                     VIVANTE_NAMESPACE ".vxcTensorLogical_or_int8"
#define VX_KERNEL_NAME_LOGICAL_OR_UINT8                    VIVANTE_NAMESPACE ".vxcTensorLogical_or_uint8"
#define VX_KERNEL_NAME_LOGICAL_OR_FP16                     VIVANTE_NAMESPACE ".vxcTensorLogical_or_fp16"
#define VX_KERNEL_NAME_LOGICAL_AND_INT16                   VIVANTE_NAMESPACE ".vxcTensorLogical_and_int16"
#define VX_KERNEL_NAME_LOGICAL_AND_INT8                    VIVANTE_NAMESPACE ".vxcTensorLogical_and_int8"
#define VX_KERNEL_NAME_LOGICAL_AND_UINT8                   VIVANTE_NAMESPACE ".vxcTensorLogical_and_uint8"
#define VX_KERNEL_NAME_LOGICAL_AND_FP16                    VIVANTE_NAMESPACE ".vxcTensorLogical_and_fp16"
#define VX_KERNEL_NAME_SELECT_UINT8                        VIVANTE_NAMESPACE ".vxcTensorSelect_uint8"
#define VX_KERNEL_NAME_SELECT_BOOL_INT8                    VIVANTE_NAMESPACE ".vxcTensorSelect_bool_int8"
#define VX_KERNEL_NAME_SELECT_BOOL_INT16                   VIVANTE_NAMESPACE ".vxcTensorSelect_bool_int16"
#define VX_KERNEL_NAME_LSTMUNIT_ACTIVATION                 VIVANTE_NAMESPACE ".vxcLSTMUnit_Activation_SW"
#define VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_FP16     VIVANTE_NAMESPACE ".vxcTensorAddMeanStdNorm_fp16"
#define VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_U8_FP16  VIVANTE_NAMESPACE ".vxcTensorAddMeanStdNorm_u8_fp16"
#define VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_I16_FP16 VIVANTE_NAMESPACE ".vxcTensorAddMeanStdNorm_i16_fp16"
#define VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_I16_FP16 VIVANTE_NAMESPACE ".vxcTensorAddMeanStdNorm_i16_fp16"
#define VX_KERNEL_NAME_STACK                               VIVANTE_NAMESPACE ".vxcStack"
//! clip kernel
#define VX_KERNEL_NAME_CLIP_F16TOF16_2D                    VIVANTE_NAMESPACE ".vxcTensorClip_F16toF16_2D"
#define VX_KERNEL_NAME_CLIP_F16TOF16                       VIVANTE_NAMESPACE ".vxcTensorClip_F16toF16"
#define VX_KERNEL_NAME_CLIP_F16TOI16_2D                    VIVANTE_NAMESPACE ".vxcTensorClip_F16toI16_2D"
#define VX_KERNEL_NAME_CLIP_F16TOI16                       VIVANTE_NAMESPACE ".vxcTensorClip_F16toI16"
#define VX_KERNEL_NAME_CLIP_F16TOI8_2D                     VIVANTE_NAMESPACE ".vxcTensorClip_F16toI8_2D"
#define VX_KERNEL_NAME_CLIP_F16TOI8                        VIVANTE_NAMESPACE ".vxcTensorClip_F16toI8"
#define VX_KERNEL_NAME_CLIP_F16TOU8_2D                     VIVANTE_NAMESPACE ".vxcTensorClip_F16toU8_2D"
#define VX_KERNEL_NAME_CLIP_F16TOU8                        VIVANTE_NAMESPACE ".vxcTensorClip_F16toU8"
#define VX_KERNEL_NAME_CLIP_U8TOF16_2D                     VIVANTE_NAMESPACE ".vxcTensorClip_U8toF16_2D"
#define VX_KERNEL_NAME_CLIP_U8TOF16                        VIVANTE_NAMESPACE ".vxcTensorClip_U8toF16"
#define VX_KERNEL_NAME_CLIP_I8TOF16_2D                     VIVANTE_NAMESPACE ".vxcTensorClip_I8toF16_2D"
#define VX_KERNEL_NAME_CLIP_I8TOF16                        VIVANTE_NAMESPACE ".vxcTensorClip_I8toF16"
#define VX_KERNEL_NAME_CLIP_I16TOF16_2D                    VIVANTE_NAMESPACE ".vxcTensorClip_I16toF16_2D"
#define VX_KERNEL_NAME_CLIP_I16TOF16                       VIVANTE_NAMESPACE ".vxcTensorClip_I16toF16"
#define VX_KERNEL_NAME_CLIP_I16TOI16_2D                    VIVANTE_NAMESPACE ".vxcTensorClip_I16toI16_2D"
#define VX_KERNEL_NAME_CLIP_I16TOI16                       VIVANTE_NAMESPACE ".vxcTensorClip_I16toI16"
#define VX_KERNEL_NAME_CLIP_I8TOI8_2D                      VIVANTE_NAMESPACE ".vxcTensorClip_I8toI8_2D"
#define VX_KERNEL_NAME_CLIP_I8TOI8                         VIVANTE_NAMESPACE ".vxcTensorClip_I8toI8"
#define VX_KERNEL_NAME_CLIP_U8TOU8_2D                      VIVANTE_NAMESPACE ".vxcTensorClip_U8toU8_2D"
#define VX_KERNEL_NAME_CLIP_U8TOU8                         VIVANTE_NAMESPACE ".vxcTensorClip_U8toU8"
//! pre process gray kernel
#define VX_KERNEL_NAME_UNSTACK                             VIVANTE_NAMESPACE ".vxcUnstack"
#define VX_KERNEL_NAME_ADDN                                VIVANTE_NAMESPACE ".vxcAddn"
#define VX_KERNEL_NAME_EXTRA_ENDING_I16                    VIVANTE_NAMESPACE ".vxcExtra_ending_i16"
#define VX_KERNEL_NAME_EXTRA_ENDING_I8                     VIVANTE_NAMESPACE ".vxcExtra_ending_i8"
#define VX_KERNEL_NAME_EXTRA_ENDING_U8                     VIVANTE_NAMESPACE ".vxcExtra_ending_u8"
#define VX_KERNEL_NAME_TOPK                                VIVANTE_NAMESPACE ".vxcTopk"
#define VX_KERNEL_NAME_LOGICAL_NOT_INT8                    VIVANTE_NAMESPACE ".vxcLogical_not_i8"
#define VX_KERNEL_NAME_LOGICAL_NOT_INT16                   VIVANTE_NAMESPACE ".vxcLogical_not_i16"
#define VX_KERNEL_NAME_LOGICAL_NOT_UINT8                   VIVANTE_NAMESPACE ".vxcLogical_not_u8"
#define VX_KERNEL_NAME_ROI_ALIGN                           VIVANTE_NAMESPACE ".vxcRoi_align"
#define VX_KERNEL_NAME_HEATMAP_MAX_KEYPOINT                VIVANTE_NAMESPACE ".vxcHeatmap_max_keypoint"
#define VX_KERNEL_NAME_AXIS_ALIGNED_BBOX_TRANSFORM         VIVANTE_NAMESPACE ".vxcAxis_aligned_bbox_transform"
#define VX_KERNEL_NAME_BOX_WITH_NMS_LIMIT                  VIVANTE_NAMESPACE ".vxcBox_with_nms_limit"
#define VX_KERNEL_NAME_GENERATE_PROPOSALS                  VIVANTE_NAMESPACE ".vxcGenerate_proposals"
#define VX_KERNEL_NAME_DETECTION_POSTPROCESS               VIVANTE_NAMESPACE ".vxcDetection_postprocess"
#define VX_KERNEL_NAME_RANDOM_MULTINOMIAL                  VIVANTE_NAMESPACE ".vxcRandom_multinomial"
#define VX_KERNEL_NAME_RANDOM_GENERATE                     VIVANTE_NAMESPACE ".vxcRandom_generate"
#define VX_KERNEL_NAME_RANDOM_SUM_FP16                     VIVANTE_NAMESPACE ".vxcRandom_sum_fp16"
#define VX_KERNEL_NAME_RANDOM_SUM_FP32                     VIVANTE_NAMESPACE ".vxcRandom_sum_fp32"

//! reducemax kernel
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOF16          VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOI16          VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toI16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOI8           VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toI8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOU8           VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toU8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOF16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOI16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOI8_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toI8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_F16TOU8_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis0_F16toU8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I16TOI16          VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I16toI16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I16TOF16          VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I16toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I16TOI16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I16TOF16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I8TOI8            VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I8toI8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I8TOF16           VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I8toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_I8TOF16_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis0_I8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_U8TOU8            VIVANTE_NAMESPACE ".vxcReducemaxAxis0_U8toU8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_U8TOF16           VIVANTE_NAMESPACE ".vxcReducemaxAxis0_U8toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_U8TOU8_2D         VIVANTE_NAMESPACE ".vxcReducemaxAxis0_U8toU8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI0_U8TOF16_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis0_U8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOF16          VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOI16          VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toI16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOI8           VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toI8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOU8           VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toU8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOF16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOI16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOI8_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toI8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_F16TOU8_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis1_F16toU8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I16TOI16          VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I16toI16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I16TOF16          VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I16toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I16TOI16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I16TOF16_2D       VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I8TOI8            VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I8toI8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I8TOF16           VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I8toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_I8TOF16_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis1_I8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_U8TOU8            VIVANTE_NAMESPACE ".vxcReducemaxAxis1_U8toU8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_U8TOF16           VIVANTE_NAMESPACE ".vxcReducemaxAxis1_U8toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_U8TOU8_2D         VIVANTE_NAMESPACE ".vxcReducemaxAxis1_U8toU8_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI1_U8TOF16_2D        VIVANTE_NAMESPACE ".vxcReducemaxAxis1_U8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_F16TOF16          VIVANTE_NAMESPACE ".vxcReducemaxAxis2_F16toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_F16TOI16          VIVANTE_NAMESPACE ".vxcReducemaxAxis2_F16toI16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_F16TOI8           VIVANTE_NAMESPACE ".vxcReducemaxAxis2_F16toI8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_F16TOU8           VIVANTE_NAMESPACE ".vxcReducemaxAxis2_F16toU8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_I16TOI16          VIVANTE_NAMESPACE ".vxcReducemaxAxis2_I16toI16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_I16TOF16          VIVANTE_NAMESPACE ".vxcReducemaxAxis2_I16toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_I8TOI8            VIVANTE_NAMESPACE ".vxcReducemaxAxis2_I8toI8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_I8TOF16           VIVANTE_NAMESPACE ".vxcReducemaxAxis2_I8toF16"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_U8TOU8            VIVANTE_NAMESPACE ".vxcReducemaxAxis2_U8toU8"
#define VX_KERNEL_NAME_REDUCEMAX_AXI2_U8TOF16           VIVANTE_NAMESPACE ".vxcReducemaxAxis2_U8toF16"
//! reducemin kernel
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOF16          VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOI16          VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toI16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOI8           VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toI8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOU8           VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toU8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOI8_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toI8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_F16TOU8_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis0_F16toU8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I16TOI16          VIVANTE_NAMESPACE ".vxcReduceminAxis0_I16toI16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I16TOF16          VIVANTE_NAMESPACE ".vxcReduceminAxis0_I16toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis0_I16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis0_I16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceminAxis0_I8toI8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I8TOF16           VIVANTE_NAMESPACE ".vxcReduceminAxis0_I8toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceminAxis0_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_I8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis0_I8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_U8TOU8            VIVANTE_NAMESPACE ".vxcReduceminAxis0_U8toU8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_U8TOF16           VIVANTE_NAMESPACE ".vxcReduceminAxis0_U8toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_U8TOU8_2D         VIVANTE_NAMESPACE ".vxcReduceminAxis0_U8toU8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI0_U8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis0_U8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOF16          VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOI16          VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toI16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOI8           VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toI8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOU8           VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toU8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOI8_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toI8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_F16TOU8_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis1_F16toU8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I16TOI16          VIVANTE_NAMESPACE ".vxcReduceminAxis1_I16toI16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I16TOF16          VIVANTE_NAMESPACE ".vxcReduceminAxis1_I16toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis1_I16toI16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceminAxis1_I16toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceminAxis1_I8toI8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I8TOF16           VIVANTE_NAMESPACE ".vxcReduceminAxis1_I8toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceminAxis1_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_I8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis1_I8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_U8TOU8            VIVANTE_NAMESPACE ".vxcReduceminAxis1_U8toU8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_U8TOF16           VIVANTE_NAMESPACE ".vxcReduceminAxis1_U8toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_U8TOU8_2D         VIVANTE_NAMESPACE ".vxcReduceminAxis1_U8toU8_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI1_U8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceminAxis1_U8toF16_2D"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_F16TOF16          VIVANTE_NAMESPACE ".vxcReduceminAxis2_F16toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_F16TOI16          VIVANTE_NAMESPACE ".vxcReduceminAxis2_F16toI16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_F16TOI8           VIVANTE_NAMESPACE ".vxcReduceminAxis2_F16toI8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_F16TOU8           VIVANTE_NAMESPACE ".vxcReduceminAxis2_F16toU8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_I16TOI16          VIVANTE_NAMESPACE ".vxcReduceminAxis2_I16toI16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_I16TOF16          VIVANTE_NAMESPACE ".vxcReduceminAxis2_I16toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceminAxis2_I8toI8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_I8TOF16           VIVANTE_NAMESPACE ".vxcReduceminAxis2_I8toF16"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_U8TOU8            VIVANTE_NAMESPACE ".vxcReduceminAxis2_U8toU8"
#define VX_KERNEL_NAME_REDUCEMIN_AXI2_U8TOF16           VIVANTE_NAMESPACE ".vxcReduceminAxis2_U8toF16"
//! reduceprod kernel
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOF16          VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOI16          VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toI16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOI8           VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toI8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOU8           VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toU8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toI16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOI8_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toI8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_F16TOU8_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis0_F16toU8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I16TOI16          VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I16toI16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I16TOF16          VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I16toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I16toI16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I16toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I8toI8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I8TOF16           VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I8toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_I8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis0_I8toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_U8TOU8            VIVANTE_NAMESPACE ".vxcReduceProdAxis0_U8toU8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_U8TOF16           VIVANTE_NAMESPACE ".vxcReduceProdAxis0_U8toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_U8TOU8_2D         VIVANTE_NAMESPACE ".vxcReduceProdAxis0_U8toU8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_U8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis0_U8toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_BF16TOBF16        VIVANTE_NAMESPACE ".vxcReduceProdAxis0_BF16toBF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI0_BF16TOBF16_2D     VIVANTE_NAMESPACE ".vxcReduceProdAxis0_BF16toBF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOF16          VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOI16          VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toI16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOI8           VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toI8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOU8           VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toU8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toI16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOI8_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toI8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_F16TOU8_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis1_F16toU8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I16TOI16          VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I16toI16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I16TOF16          VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I16toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I16TOI16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I16toI16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I16TOF16_2D       VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I16toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I8toI8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I8TOF16           VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I8toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_I8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis1_I8toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_U8TOU8            VIVANTE_NAMESPACE ".vxcReduceProdAxis1_U8toU8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_U8TOF16           VIVANTE_NAMESPACE ".vxcReduceProdAxis1_U8toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_U8TOU8_2D         VIVANTE_NAMESPACE ".vxcReduceProdAxis1_U8toU8_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_U8TOF16_2D        VIVANTE_NAMESPACE ".vxcReduceProdAxis1_U8toF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_BF16TOBF16        VIVANTE_NAMESPACE ".vxcReduceProdAxis1_BF16toBF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI1_BF16TOBF16_2D     VIVANTE_NAMESPACE ".vxcReduceProdAxis1_BF16toBF16_2D"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_F16TOF16          VIVANTE_NAMESPACE ".vxcReduceProdAxis2_F16toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_F16TOI16          VIVANTE_NAMESPACE ".vxcReduceProdAxis2_F16toI16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_F16TOI8           VIVANTE_NAMESPACE ".vxcReduceProdAxis2_F16toI8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_F16TOU8           VIVANTE_NAMESPACE ".vxcReduceProdAxis2_F16toU8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_I16TOI16          VIVANTE_NAMESPACE ".vxcReduceProdAxis2_I16toI16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_I16TOF16          VIVANTE_NAMESPACE ".vxcReduceProdAxis2_I16toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceProdAxis2_I8toI8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_I8TOF16           VIVANTE_NAMESPACE ".vxcReduceProdAxis2_I8toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_U8TOU8            VIVANTE_NAMESPACE ".vxcReduceProdAxis2_U8toU8"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_U8TOF16           VIVANTE_NAMESPACE ".vxcReduceProdAxis2_U8toF16"
#define VX_KERNEL_NAME_REDUCEPROD_AXI2_BF16TOBF16        VIVANTE_NAMESPACE ".vxcReduceProdAxis2_BF16toBF16"
//! reduceall kernel
#define VX_KERNEL_NAME_REDUCEALL_AXI0_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceallAxis0_I8toI8"
#define VX_KERNEL_NAME_REDUCEALL_AXI0_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceallAxis0_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEALL_AXI1_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceallAxis1_I8toI8"
#define VX_KERNEL_NAME_REDUCEALL_AXI1_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceallAxis1_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEALL_AXI2_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceallAxis2_I8toI8"
//! reduceany kernel
#define VX_KERNEL_NAME_REDUCEANY_AXI0_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceanyAxis0_I8toI8"
#define VX_KERNEL_NAME_REDUCEANY_AXI0_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceanyAxis0_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEANY_AXI1_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceanyAxis1_I8toI8"
#define VX_KERNEL_NAME_REDUCEANY_AXI1_I8TOI8_2D         VIVANTE_NAMESPACE ".vxcReduceanyAxis1_I8toI8_2D"
#define VX_KERNEL_NAME_REDUCEANY_AXI2_I8TOI8            VIVANTE_NAMESPACE ".vxcReduceanyAxis2_I8toI8"
//! bilinear
#define VX_KERNEL_NAME_RESIZE_INTERNAL_I8TOI8_UP        VIVANTE_NAMESPACE ".vxcResize_I8toI8_up"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_I8TOI8           VIVANTE_NAMESPACE ".vxcResize_I8toI8"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_I16TOI16_UP      VIVANTE_NAMESPACE ".vxcResize_I16toI16_up"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_I16TOI16         VIVANTE_NAMESPACE ".vxcResize_I16toI16"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_U8TOF16          VIVANTE_NAMESPACE ".vxcResize_U8toF16"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_U8TOU8_UP        VIVANTE_NAMESPACE ".vxcResize_U8toU8_up"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_U8TOU8           VIVANTE_NAMESPACE ".vxcResize_U8toU8"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_F16TOF16_UP      VIVANTE_NAMESPACE ".vxcResize_F16toF16_up"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_F16TOF16         VIVANTE_NAMESPACE ".vxcResize_F16toF16"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_F16TOU8          VIVANTE_NAMESPACE ".vxcResize_F16toU8"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_BF16TOBF16_UP    VIVANTE_NAMESPACE ".vxcResize_BF16toBF16_up"
#define VX_KERNEL_NAME_RESIZE_INTERNAL_BF16TOBF16       VIVANTE_NAMESPACE ".vxcResize_BF16toBF16"
//! nearest resize
#define VX_KERNEL_NAME_NEAREST_INTERNAL_I16TOI16        VIVANTE_NAMESPACE ".vxcResize_nearest_I16toI16"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_I16TOI16_OP     VIVANTE_NAMESPACE ".vxcResize_nearest_I16toI16_op"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_F16TOF16        VIVANTE_NAMESPACE ".vxcResize_nearest_F16toF16"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_F16TOF16_OP     VIVANTE_NAMESPACE ".vxcResize_nearest_F16toF16_op"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_U8TOU8          VIVANTE_NAMESPACE ".vxcResize_nearest_U8toU8"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_U8TOU8_OP       VIVANTE_NAMESPACE ".vxcResize_nearest_U8toU8_op"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_I8TOI8          VIVANTE_NAMESPACE ".vxcResize_nearest_I8toI8"
#define VX_KERNEL_NAME_NEAREST_INTERNAL_I8TOI8_OP       VIVANTE_NAMESPACE ".vxcResize_nearest_I8toI8_op"

/*! \brief The list of Example Kernels.
 * \ingroup group_xyz_ext
 */
//! [KERNEL ENUM]
enum vx_kernel_libnnext_ext_e
{
    /*! \brief The Example Kernel */
    VX_KERNEL_ENUM_LIBNNEXT             =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LIBNNEXT_OFFSET,
    VX_KERNEL_ENUM_PERMUTECWH           =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PREMUTE_OFFSET,
    VX_KERNEL_ENUM_PERMUTECHW           =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PREMUTE_OFFSET + 1,
    VX_KERNEL_ENUM_PRIORBOX             =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PRIORBOX_OFFSET,
    VX_KERNEL_ENUM_FLATTEN              =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_FLATTEN_OFFSET,
    VX_KERNEL_ENUM_L2NORMALIZESCALE     =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_L2NORMALIZESCALE_OFFSET,
    VX_KERNEL_ENUM_L2NORMSCALE_SUMRSQRT =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_L2NORMALIZESCALE_OFFSET + 1,
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_L2NORMALIZESCALE_OFFSET + 2,
    VX_KERNEL_ENUM_PARAMETRICRELU       =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PARAMETRICRELU_OFFSET,
    VX_KERNEL_ENUM_PREBBOX              =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PREBBOX_OFFSET,
    VX_KERNEL_ENUM_ADD_RELU_KERNEL      =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ADD_RELU_KERNEL_OFFSET,
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX  =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_POOLING_WITH_ARGMAX_OFFSET,
    VX_KERNEL_ENUM_UNPOOLING            =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_UNPOOLING_OFFSET,
    VX_KERNEL_ENUM_ARGMAX               =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ARGMAX_OFFSET,
    VX_KERNEL_ENUM_ALEXNET_GEMM         =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ALEXNET_GEMM_OFFSET,
    VX_KERNEL_ENUM_IMG2COL_DILATED      =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_IMG2COL_DILATED_OFFSET,
    VX_KERNEL_ENUM_IMG2COL_DILATED_INT8 =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_IMG2COL_DILATED_INT8_OFFSET,
    VX_KERNEL_ENUM_ALEXNET_GEMM_INT8    =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ALEXNET_GEMM_INT8_OFFSET,
    VX_KERNEL_ENUM_MAXIMUM          =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_ENUM_FULLYCONNECTED_AXIS2 =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_FULLYCONNECTED_AXIS2,
    VX_KERNEL_ENUM_TENSORCROP_INT16     =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORCROP_INT16,
    VX_KERNEL_ENUM_TENSORCROP_INT8      =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORCROP_INT8,
    VX_KERNEL_ENUM_TENSORCROP_INT16_FP16 =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORCROP_INT16_FP16,
    VX_KERNEL_ENUM_DROPOUT              =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_DROPOUT,
    VX_KERNEL_ENUM_SHUFFLECHANNEL       =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_ENUM_RESIZE               =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RESIZE,
    VX_KERNEL_ENUM_REVERSE              =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REVERSE,
    VX_KERNEL_ENUM_RESIZE_16BITS_DOWNSAMPLE_QUARTER =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RESIZE_16BITS_DOWNSAMPLE_QUARTER,
    VX_KERNEL_ENUM_RESIZE_8BITS_DOWNSAMPLE_QUARTER =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RESIZE_8BITS_DOWNSAMPLE_QUARTER,
    VX_KERNEL_ENUM_SCALE                = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SCALE,
    VX_KERNEL_ENUM_TENSORREVERSE        =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORREVERSE,
    VX_KERNEL_ENUM_TENSORELU            =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORELU_OFFSET,
    VX_KERNEL_ENUM_SPACE2BATCH          =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SPACE2BATCH,
    VX_KERNEL_ENUM_BATCH2SPACE          =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_BATCH2SPACE,
    VX_KERNEL_ENUM_SPACE2DEPTH          =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SPACE2DEPTH,
    VX_KERNEL_ENUM_IMAGEPROCESS         =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_IMAGEPROCESS,
    VX_KERNEL_ENUM_SCALETOTENSOR        =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR    =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_ENUM_GEMM                 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_GEMM,
    VX_KERNEL_ENUM_LAYERNORM            = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LAYERNORM,
    VX_KERNEL_ENUM_LAYERNORM_FP16TOU8   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LAYERNORMFP16TOU8_OFFSET,
    VX_KERNEL_ENUM_REDUCE               = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REDUCE,
    VX_KERNEL_ENUM_INSTANCENORM         =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_ENUM_TENSORSTACKCONCAT    =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORSTACKCONCAT,
    VX_KERNEL_ENUM_TENSORSTACKCONCAT8BITS =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSORSTACKCONCAT8BITS_OFFSET,
    VX_KERNEL_ENUM_SIGNALFRAME          =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_ENUM_RELATIONALOPS        =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_ENUM_POW        =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_POW,
    VX_KERNEL_ENUM_FLOORDIV        =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER  =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_ENUM_LOGICAL_OPS          = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LOGICAL_OPS,
    VX_KERNEL_ENUM_SELECT               = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SELECT,
    VX_KERNEL_ENUM_LSTMUNIT_ACTIVATION  =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LSTMUNIT_ACTIVATION,
    VX_KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM,
    VX_KERNEL_ENUM_STACK                = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_STACK,
    VX_KERNEL_ENUM_NEG                  = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_NEG,
    VX_KERNEL_ENUM_TENSOR_EXP           = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_EXP,
    VX_KERNEL_ENUM_CLIP                 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_CLIP,
    VX_KERNEL_ENUM_PRE_PROCESS_GRAY     =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PRE_PROCESS_GRAY,
    VX_KERNEL_ENUM_UNSTACK              = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_UNSTACK,
    VX_KERNEL_ENUM_PRE_PROCESS_RGB      =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PRE_PROCESS_RGB,
    VX_KERNEL_ENUM_ADDN                 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ADDN,
    VX_KERNEL_ENUM_PRE_PROCESS_YUV420   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PRE_PROCESS_YUV420,
    VX_KERNEL_ENUM_CONV2D               = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_CONV2D,
    VX_KERNEL_ENUM_EXTRA_ENDING         =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_EXTRA_ENDING,
    VX_KERNEL_ENUM_GATHER               = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_GATHER,
    VX_KERNEL_ENUM_TILE                 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TILE,
    VX_KERNEL_ENUM_TOPK                 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_TOPK,
    VX_KERNEL_ENUM_PRE_PROCESS_BGRA     =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PRE_PROCESS_BGRA,
    VX_KERNEL_ENUM_LOGICAL_NOT          =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LOGICAL_NOT,
    VX_KERNEL_ENUM_TENSOR_SIN           = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_SIN,
    VX_KERNEL_ENUM_TENSOR_LOG           = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LOG,
    VX_KERNEL_ENUM_ARGMIN               = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ARGMIN,
    VX_KERNEL_ENUM_ROI_ALIGN            = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_ROI_ALIGN,
    VX_KERNEL_ENUM_HEATMAP_MAX_KEYPOINT =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_HEATMAP_MAX_KEYPOINT,
    VX_KERNEL_ENUM_AXIS_ALIGNED_BBOX_TRANSFORM =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_AXIS_ALIGNED_BBOX_TRANSFORM,
    VX_KERNEL_ENUM_BOX_WITH_NMS_LIMIT   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_BOX_WITH_NMS_LIMIT,
    VX_KERNEL_ENUM_GENERATE_PROPOSALS   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_GENERATE_PROPOSALS,
    VX_KERNEL_ENUM_DETECTION_POSTPROCESS =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_DETECTION_POSTPROCESS,
    VX_KERNEL_ENUM_RANDOM_MULTINOMIAL   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RANDOM_MULTINOMIAL,
    VX_KERNEL_ENUM_LOG_SOFTMAX          = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_LOG_SOFTMAX,
    VX_KERNEL_ENUM_RELU_KERAS_INTERNAL  =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RELU_KERAS_INTERNAL,
    VX_KERNEL_ENUM_DECONV2D               = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_DECONV2D,
    VX_KERNEL_ENUM_REDUCEMAX_INTERNAL   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REDUCEMAX_INTERNAL,
    VX_KERNEL_ENUM_REDUCEMIN_INTERNAL   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REDUCEMIN_INTERNAL,
    VX_KERNEL_ENUM_REDUCEPROD_INTERNAL  =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REDUCEPROD_INTERNAL,
    VX_KERNEL_ENUM_REDUCEALL_INTERNAL   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REDUCEALL_INTERNAL,
    VX_KERNEL_ENUM_REDUCEANY_INTERNAL   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_REDUCEANY_INTERNAL,
    VX_KERNEL_ENUM_RESIZE_INTERNAL      =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RESIZE_INTERNAL,
    VX_KERNEL_ENUM_RESIZE_NEAREST_INTERNAL =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_RESIZE_NEAREST_INTERNAL,
    VX_KERNEL_ENUM_PRE_PROCESS_YUV444   =
            VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_PRE_PROCESS_YUV444,
    // up to 0xFFF kernel enums can be created.
};



/* Assigned from Khronos, for custom */
#define VX_LIBRARY_CUSTOM (0x4)
enum vx_kernel_custom_id_e
{
    _VX_CLIENT_ID_START = VX_KERNEL_BASE( VX_ID_DEFAULT, VX_LIBRARY_CUSTOM ),
#define DEF_OP( name )     VX_CLIENT_ID_##name,
    #include "custom/custom_ops.def"
#undef DEF_OP
};
#define VX_KERNEL_ID( name ) VX_CLIENT_ID_##name

#ifndef gvxOBJ_CHECK
#define gvxOBJ_CHECK(ref) \
    do \
    { \
        status = vxGetStatus((vx_reference)ref); \
        if (ref == 0 || status != VX_SUCCESS) \
        { \
            printf("Obj ERROR: status=%d @ %s(%d)\n", status, __FUNCTION__, __LINE__); \
        } \
    } \
    while (0)
#endif
#ifndef gvxSTATUS_CHECK
#define gvxSTATUS_CHECK(status) \
    do \
    { \
        if (status != VX_SUCCESS) \
        { \
            printf("status ERROR: status=%d @ %s(%d)\n", status, __FUNCTION__, __LINE__); \
        } \
    } \
    while (0)
#endif

#ifdef __cplusplus
}
#endif

#endif

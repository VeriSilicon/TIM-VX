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
#include "type_utils.h"

namespace tim {
namespace vx {
vsi_nn_type_e TranslateDataType(DataType dtype) {
  switch (dtype) {
    case DataType::INT8:
      return VSI_NN_TYPE_INT8;
    case DataType::UINT8:
      return VSI_NN_TYPE_UINT8;
    case DataType::INT16:
      return VSI_NN_TYPE_INT16;
    case DataType::UINT16:
      return VSI_NN_TYPE_UINT16;
    case DataType::INT32:
      return VSI_NN_TYPE_INT32;
    case DataType::UINT32:
      return VSI_NN_TYPE_UINT32;
    case DataType::FLOAT16:
      return VSI_NN_TYPE_FLOAT16;
    case DataType::FLOAT32:
      return VSI_NN_TYPE_FLOAT32;
    case DataType::BOOL8:
      return VSI_NN_TYPE_BOOL8;
    default:
      break;
  }
  return VSI_NN_TYPE_FLOAT16;
}

vsi_nn_qnt_type_e TranslateQuantType(QuantType qtype) {
  switch (qtype) {
    case QuantType::NONE:
      return VSI_NN_QNT_TYPE_NONE;
    case QuantType::ASYMMETRIC:
      return VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    case QuantType::SYMMETRIC_PER_CHANNEL:
      return VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
    default:
      break;
  }
  return VSI_NN_QNT_TYPE_NONE;
}

vsi_nn_pad_e TranslatePadType(PadType pad) {
  switch (pad) {
    case PadType::AUTO:
      return VSI_NN_PAD_AUTO;
    case PadType::VALID:
      return VSI_NN_PAD_VALID;
    case PadType::SAME:
      return VSI_NN_PAD_SAME;

    default:
      break;
  }
  return VSI_NN_PAD_AUTO;
}

vsi_enum TranslatePoolType(PoolType type) {
  switch (type) {
    case PoolType::MAX:
      return VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    case PoolType::AVG:
      return VX_CONVOLUTIONAL_NETWORK_POOLING_AVG;
    case PoolType::L2:
      return VX_CONVOLUTIONAL_NETWORK_POOLING_L2;
    case PoolType::AVG_ANDROID:
      return VX_CONVOLUTIONAL_NETWORK_POOLING_AVG_ANDROID;

    default:
      break;
  }
  return VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
}

vsi_nn_round_type_e TranslateRoundType(RoundType type) {
  switch (type) {
    case RoundType::CEILING:
      return VSI_NN_ROUND_CEIL;
    case RoundType::FLOOR:
      return VSI_NN_ROUND_FLOOR;

    default:
      break;
  }
  return VSI_NN_ROUND_CEIL;
}

vsi_enum TranslateOverflowPolicy(OverflowPolicy type) {
  switch (type) {
    case OverflowPolicy::WRAP:
      return VX_CONVERT_POLICY_WRAP;
    case OverflowPolicy::SATURATE:
      return VX_CONVERT_POLICY_SATURATE;
    default:
      break;
  }
  return VX_CONVERT_POLICY_SATURATE;
}

vsi_enum TranslateRoundingPolicy(RoundingPolicy type) {
  switch (type) {
    case RoundingPolicy::TO_ZERO:
      return VX_ROUND_POLICY_TO_ZERO;
    case RoundingPolicy::RTNE:
      return VX_ROUND_POLICY_TO_NEAREST_EVEN;
    default:
      break;
  }
  return VX_ROUND_POLICY_TO_NEAREST_EVEN;
}

vsi_enum TranslateDownScaleSizeRounding(RoundType type) {
  switch (type) {
    case RoundType::FLOOR:
      return VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
    case RoundType::CEILING:
      return VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING;
    default:
      break;
  }
  return VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
}

vsi_enum TranslateResizeType(ResizeType type) {
  switch (type) {
    case ResizeType::NEAREST_NEIGHBOR:
      return VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR;
    case ResizeType::BILINEAR:
      return VSI_NN_INTERPOLATION_BILINEAR;
    case ResizeType::AREA:
      return VSI_NN_INTERPOLATION_AREA;
  }
  return VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR;
}

vx_bool_e ToVxBool(bool val) { return val ? vx_true_e : vx_false_e; }

}  // namespace vx
}  // namespace tim

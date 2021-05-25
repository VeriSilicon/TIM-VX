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
#ifndef TIM_VX_TYPES_H_
#define TIM_VX_TYPES_H_

namespace tim {
namespace vx {

enum class DataType {
  UNKNOWN,
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  FLOAT16,
  FLOAT32,
  BOOL8
};

enum class QuantType { NONE, ASYMMETRIC, SYMMETRIC_PER_CHANNEL };

enum TensorAttribute {
  CONSTANT = 1 << 0,
  TRANSIENT = 1 << 1,
  VARIABLE = 1 << 2,
  INPUT = 1 << 3,
  OUTPUT = 1 << 4
};

enum class PadType { NONE = -1, AUTO, VALID, SAME };

enum class PoolType { MAX, AVG, L2, AVG_ANDROID };

enum class RoundType { CEILING, FLOOR };

enum class OverflowPolicy { WRAP, SATURATE };

enum class RoundingPolicy { TO_ZERO, RTNE };

enum class ResizeType { NEAREST_NEIGHBOR, BILINEAR, AREA };

enum class DataLayout {
  WHCN,
  CWHN,
  ANY,
  IcWHOc /*TF*/,
  OcIcWH /*TVM*/,
  WHIcOc /*TIM-VX default*/
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_TYPES_H_ */

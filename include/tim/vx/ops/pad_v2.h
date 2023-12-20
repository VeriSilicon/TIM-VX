/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
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
#ifndef TIM_VX_OPERATION_PADV2_H_
#define TIM_VX_OPERATION_PADV2_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## PadV2
 *
 * Pads a tensor.
 *
 * - const_val : the float value to pad.
 * - pad_mode : the mode of pad.
 * - front_size : Add pad values to the left and top.
 * - back_size : Add pad values to the right and bottom.
 */

class PadV2 : public BuiltinOp {
 public:
  typedef enum {
    // signature
    PAD_MODE_CONSTANT,
    PAD_MODE_EDGE,
    PAD_MODE_SYMMETRIC,
    PAD_MODE_REFLECT,
  } pad_mode_type;

  PadV2(Graph* graph, const std::vector<uint32_t>& front_size,
           const std::vector<uint32_t>& back_size, float const_val);
  PadV2(Graph* graph, const std::vector<uint32_t>& front_size,
      const std::vector<uint32_t>& back_size, float const_val,
      pad_mode_type pad_mode);

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  std::vector<uint32_t> front_size_;
  std::vector<uint32_t> back_size_;
  float const_val_;
  pad_mode_type pad_mode_;
};
}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif
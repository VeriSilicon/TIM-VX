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
#include "tim/vx/ops/pad_v2.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

PadV2::PadV2(Graph* graph, const std::vector<uint32_t>& front_size,
         const std::vector<uint32_t>& back_size, float const_val)
    : PadV2(graph, front_size, back_size, const_val, PAD_MODE_CONSTANT) {}

PadV2::PadV2(Graph* graph, const std::vector<uint32_t>& front_size,
         const std::vector<uint32_t>& back_size, float const_val,
         pad_mode_type pad_mode)
    : BuiltinOp(graph, VSI_NN_OP_PAD2),
      front_size_(front_size),
      back_size_(back_size),
      const_val_(const_val),
      pad_mode_(pad_mode) {
  this->impl()->node()->nn_param.pad2.front_size = front_size_.data();
  this->impl()->node()->nn_param.pad2.back_size = back_size_.data();
  this->impl()->node()->nn_param.pad2.dim_num = front_size_.size();
  if (pad_mode_ == PAD_MODE_CONSTANT) {
    this->impl()->node()->nn_param.pad2.const_val = const_val_;
  }
  this->impl()->node()->nn_param.pad2.mode = (vsi_nn_pad_mode_e)pad_mode_;
}

std::shared_ptr<Operation> PadV2::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<PadV2>(this->front_size_, this->back_size_,
                                     this->const_val_, this->pad_mode_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
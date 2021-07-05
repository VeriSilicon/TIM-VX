/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/maxpoolwithargmax.h"

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

MaxpoolWithArgmax::MaxpoolWithArgmax(Graph* graph, PadType padding,
                                   const std::array<uint32_t, 2>& ksize,
                                   const std::array<uint32_t, 2>& stride,
                                   RoundType round_type,
                                   DataLayout layout)
    : Operation(graph, VSI_NN_OP_POOLWITHARGMAX, 1, 2, layout),
      padding_(padding),
      ksize_(ksize),
      stride_(stride),
      round_type_(round_type) {
  this->impl()->node()->nn_param.pool.type = TranslatePoolType(PoolType::MAX);
  this->impl()->node()->nn_param.pool.round_type =
      TranslateRoundType(round_type_);
  this->impl()->node()->nn_param.pool.ksize[0] = ksize_[0];
  this->impl()->node()->nn_param.pool.ksize[1] = ksize_[1];
  this->impl()->node()->nn_param.pool.stride[0] = stride_[0];
  this->impl()->node()->nn_param.pool.stride[1] = stride_[1];
  this->impl()->node()->nn_param.pool.pad_type = TranslatePadType(padding_);
  this->SetRoundingPolicy(OverflowPolicy::SATURATE, RoundingPolicy::RTNE, round_type_);
}

std::shared_ptr<Operation> MaxpoolWithArgmax::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<MaxpoolWithArgmax>(
      this->padding_, this->ksize_, this->stride_, this->round_type_,
      this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim

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
#ifndef TIM_LAYOUT_INFER_REDUCE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_REDUCE_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/reduce.h"

#include <set>

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/vx/operation_private.h"

namespace tim {
namespace transform {
template <typename OpType>
class ReduceLayoutInfer : public OpLayoutInfer {
  public:
  ReduceLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensor) override {
    auto t_src = op_->impl()->InputsTensor()[0];
    auto pv = context_->GetPermuteVector(op_->impl()->InputsTensor()[0]);
    std::set<int32_t> unique_axis;
    std::vector<int32_t> new_axis;
    for (uint32_t i = 0; i < op_->impl()->node()->nn_param.reduce.axis_num;
         ++i) {
      int32_t axis = op_->impl()->node()->nn_param.reduce.axis[i];
      if (axis < 0) {
        axis += pv->Rank();
      }
      unique_axis.insert(axis);
      new_axis.push_back(MapAxis(pv->AsStdVec(), axis));
    }
    auto reduce = context_->infer_graph_->CreateOperation<OpType>(
        new_axis, op_->impl()->node()->nn_param.reduce.keep_dim);
    (*reduce).BindInput(context_->GetMapedTensor(t_src));
    if (op_->impl()->node()->nn_param.reduce.keep_dim) {
      auto otensor_infer = CreateOutputsTensor(pv);
      (*reduce).BindOutput(otensor_infer[0]);
      context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], pv);
    } else {
      auto out_pv = MakeShared(pv->Rank() - unique_axis.size());
      uint32_t j = 0;
      for (uint32_t i = 0; i < out_pv->Rank(); i++) {
        if (unique_axis.end() != unique_axis.find(pv->At(i))) continue;
        uint32_t cnt = 0;
        for (auto axis : unique_axis) {
          if (pv->At(i) > (uint32_t)axis) cnt++;
        }
        out_pv->At(j) = pv->At(i) - cnt;
        j++;
      }
      auto otensor_infer = CreateOutputsTensor(out_pv);
      (*reduce).BindOutput(otensor_infer[0]);
      context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], out_pv);
    }
    next_tensor.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

using ReduceMinLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceMin>;
using ReduceMaxLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceMax>;
using ReduceAnyLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceAny>;
using ReduceProdLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceProd>;
using ReduceMeanLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceMean>;
using ReduceSumLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceSum>;
}  // namespace transform
}  // namespace tim

#endif
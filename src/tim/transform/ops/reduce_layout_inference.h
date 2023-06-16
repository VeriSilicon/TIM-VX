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
#ifndef TIM_LAYOUT_INFER_REDUCE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_REDUCE_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/reduce.h"

#include <set>

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

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
    std::set<int32_t> axis_set; //Same value as new_axis, convenient for searching
    std::vector<int32_t> new_axis, pv_reduced;
    for (uint32_t i = 0; i < op_->impl()->node()->nn_param.reduce.axis_num;
         ++i) {
      int32_t axis = op_->impl()->node()->nn_param.reduce.axis[i];
      if (axis < 0) {
        axis += pv->Rank();
      }
      axis = MapAxis(pv->AsStdVec(), axis);
      axis_set.insert(axis);
      new_axis.push_back(axis);
    }
    auto reduce = context_->infer_graph_->CreateOperation<OpType>(
        new_axis, op_->impl()->node()->nn_param.reduce.keep_dim);
    (*reduce).BindInput(context_->GetMapedTensor(t_src));
    if (op_->impl()->node()->nn_param.reduce.keep_dim) {
      auto otensor_infer = CreateOutputsTensor(pv);
      (*reduce).BindOutput(otensor_infer[0]);
      context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], pv);
    } else {
      auto out_pv = MakeShared(pv->Rank() - axis_set.size());
      for (uint32_t i = 0; i < pv->Rank(); i++) {
        if (axis_set.end() != axis_set.find(i)) continue;
        pv_reduced.push_back(pv->At(i));
      }
      uint32_t j = 0;
      for (auto axis_remine : pv_reduced) {
        uint32_t cnt = 0;
        for(auto axis_reduced : axis_set) {
          if ((uint32_t)axis_remine > pv->At(axis_reduced)) cnt++;
        }
        out_pv->At(j) = (uint32_t)axis_remine - cnt;
        ++j;
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
using ReduceAllLayoutInfer = ReduceLayoutInfer<tim::vx::ops::ReduceAll>;
}  // namespace transform
}  // namespace tim

#endif
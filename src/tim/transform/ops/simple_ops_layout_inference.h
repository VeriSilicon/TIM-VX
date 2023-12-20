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
#ifndef TIM_LAYOUT_INFER_SIMMPLE_OPS_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_SIMMPLE_OPS_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/simple_operations.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {
template <typename OpType>
class SimpleOpsLayoutInfer : public OpLayoutInfer {
 public:
  SimpleOpsLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    // Transmit input pv to out pv directly for simple ops
    assert(op_->impl()->InputsTensor().size() == 1);
    auto i_src = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(i_src);
    auto out_infer = CreateOutputsTensor(input_pv);
    auto simple_op = context_->infer_graph_->CreateOperation<OpType>();
    (*simple_op)
        .BindInput(context_->GetMapedTensor(i_src))
        .BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

using DataConvertLayoutInfer = SimpleOpsLayoutInfer<vx::ops::DataConvert>;
using NegLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Neg>;
using AbsLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Abs>;
using SinLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Sin>;
// TODO(yzw): enable it when TIM-VX support 'Cos'
// using CosLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Cos>;
using ExpLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Exp>;
using LogLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Log>;
using SqrtLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Sqrt>;
using RsqrtLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Rsqrt>;
using SquareLayoutInfer = SimpleOpsLayoutInfer<vx::ops::Square>;
using LogicalNotLayoutInfer = SimpleOpsLayoutInfer<vx::ops::LogicalNot>;

}  // namespace transform
}  // namespace tim

#endif
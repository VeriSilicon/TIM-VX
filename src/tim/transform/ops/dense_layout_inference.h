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
#ifndef TIM_LAYOUT_INFER_DENSE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_DENSE_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/dense.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "direct_map_op_impl.h"

#include <memory>

namespace tim {
namespace transform {
class DenseLayoutInfer : public OpLayoutInfer {
 public:
  DenseLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_pv = context_->GetPermuteVector(input_tensors.at(0));

    if (input_tensors.at(0)->GetShape().size() > 2) {
      ReverseInputsPermuteVector();
      uint32_t input_size = input_tensors.at(1)->GetShape()[0];
      uint32_t total_input_size = 1;
      for (uint8_t i = 0; i < input_tensors.at(0)->GetShape().size(); i++) {
        total_input_size *= input_tensors.at(0)->GetShape()[i];
      }
      uint32_t input_batch = total_input_size / input_size;
      auto reshape_spec = input_tensors.at(0)->GetSpec().AsTransientSpec();
      auto reshape_output = context_->infer_graph_->CreateTensor(reshape_spec);
      std::vector<uint32_t> new_shape{input_size, input_batch};
      auto reshape_op =
          context_->infer_graph_->CreateOperation<tim::vx::ops::Reshape>(
              new_shape);
      (*reshape_op).BindInput(input_tensors.at(0));
      (*reshape_op).BindOutput(reshape_output);

      input_pv = MakeShared(new_shape.size());
      context_->UpdateTensorMap(input_tensors.at(0), reshape_output);
      context_->SetPermuteVector(input_tensors.at(0), input_pv);
    }

    for (const auto& in : input_tensors) {  // permute const weight data
      if (in->IsConstTensor() && !input_pv->IsAligned()) {
        auto perm_out = PermuteConstTensor(in, input_pv);
        context_->UpdateTensorMap(in, perm_out);
        context_->SetPermuteVector(in, input_pv);
        break;
      }
    }

    for (const auto& in : input_tensors) {  // input and weight are both inputs
      if (!in->IsConstTensor() &&
          context_->GetPermuteVector(in)->AsStdVec() != input_pv->AsStdVec()) {
        auto perm_out = InsertPermute(context_->GetMapedTensor(in), input_pv);
        context_->UpdateTensorMap(in, perm_out);
        context_->SetPermuteVector(in, input_pv);
        break;
      }
    }

    /* Because the input is larger than 2 dimensions, the reshape will be inserted, and
      the reshape will restore the default layout, so only the default pv will be used
      when inserting the reshape. Input equal to 2 dimensions may have unaligned pv. */
    int32_t axis;
    if (input_pv->IsAligned()) {
      axis = 0;
    } else {
      axis = 1;
    }
    auto infer_tensor = context_->infer_graph_->CreateTensor(
        input_tensors.at(0)->GetSpec(), input_tensors.at(0)->GetDataRef());
    auto fc =
        context_->infer_graph_->CreateOperation<tim::vx::ops::FullyConnected>(
            axis, input_tensors[1]->GetShape()[1]);

    auto out_infer = CreateOutputsTensor(input_pv);
    for (auto in : op_->impl()->InputsTensor()) {
      (*fc).BindInput(context_->GetMapedTensor(in));
    }
    (*fc).BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim
#endif
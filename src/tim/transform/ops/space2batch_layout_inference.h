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
#ifndef TIM_LAYOUT_INFER_SPACE2BATCH_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_SPACE2BATCH_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/space2batch.h"

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/vx/operation_private.h"
namespace tim {
namespace transform {
class Space2BatchLayoutInfer : public OpLayoutInfer {
 public:
  Space2BatchLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    vx::DataLayout layout = op_->impl()->layout_;
    auto required_pv = MakeShared(4);
    if (layout == vx::DataLayout::CWHN) {
      required_pv = std::make_shared<PermuteVector<4>>(kCWHN2WHCN);
    }
    auto input_tensors = op_->impl()->InputsTensor();

    auto pv = context_->GetPermuteVector(input_tensors[0]);
    auto final_pv = pv->Reverse()->Add(required_pv);
    if (!final_pv->IsAligned()) {
      auto perm_out =
          InsertPermute(context_->GetMapedTensor(input_tensors[0]), final_pv);
      context_->UpdateTensorMap(input_tensors[0], perm_out);
      context_->SetPermuteVector(input_tensors[0], required_pv);
    }

    uint32_t block_size_num =
        op_->impl()->node()->nn_param.space2batch.block_size_num;
    std::vector<int> block_size(block_size_num);
    memcpy(block_size.data(),
           op_->impl()->node()->nn_param.space2batch.block_size,
           sizeof(int) * block_size_num);
    std::vector<int> pad(4);
    memcpy(pad.data(), op_->impl()->node()->nn_param.space2batch.pad,
           sizeof(int) * 4);

    auto space2batch =
        context_->infer_graph_->CreateOperation<vx::ops::Space2Batch>(
            block_size, pad, vx::DataLayout::WHCN);
    auto out_tensor_infer = CreateOutputsTensor(required_pv);
    (*space2batch).BindInput(context_->GetMapedTensor(input_tensors[0]));
    (*space2batch).BindOutput(out_tensor_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif
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
#ifndef TIM_LAYOUT_INFER_CONV2D_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_CONV2D_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/conv2d.h"

#include "src/tim/vx/operation_private.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/transform/ops/op_layout_inference.h"

namespace tim {
namespace transform {
class Conv2dLayoutInfer : public OpLayoutInfer {
 public:
  Conv2dLayoutInfer(
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

    for (const auto& in : input_tensors) {
      std::shared_ptr<vx::Tensor> infer_tensor;
      std::shared_ptr<IPermuteVector> trans_pv;
      if (in->IsConstTensor() &&
          !(in->GetSpec().attr_ & vx::TensorAttribute::INPUT)) {
            // For bias
            if (in->GetShape().size() == 1) {
              infer_tensor = context_->infer_graph_->CreateTensor(
                  in->GetSpec(), in->GetDataRef());
              trans_pv = MakeShared(1);
            } else {
              // For input/weight
              if (!required_pv->IsAligned()) {
                auto src_conv2d = std::static_pointer_cast<vx::ops::Conv2d>(op_);
                // Support TVM Kernel Layout
                if (src_conv2d->KernelDataLayout() == vx::DataLayout::OcIcWH) {
                  trans_pv = std::make_shared<PermuteVector<4>>(kOcIcWH2WHIcOc);
                  infer_tensor = PermuteConstTensor(
                      in, trans_pv);
                } else {
                  infer_tensor = PermuteConstTensor(in, required_pv);
                  trans_pv = required_pv;
                }
              } else {
                infer_tensor = context_->infer_graph_->CreateTensor(
                    in->GetSpec(), in->GetDataRef());
                trans_pv = MakeShared(required_pv->Rank());
              }
            }
      } else {
        // For bias
        if (in->GetShape().size() == 1) {
          infer_tensor = context_->GetMapedTensor(in);
          trans_pv = MakeShared(1);
        } else {
          // For input/weight
          auto pv = context_->GetPermuteVector(in);
          auto final_pv = pv->Reverse()->Add(required_pv);
          if (!final_pv->IsAligned()) {
            infer_tensor =
                InsertPermute(context_->GetMapedTensor(in), final_pv);
            trans_pv = required_pv;
          } else {
            infer_tensor = context_->GetMapedTensor(in);
            trans_pv = pv;
          }
        }
      }
      context_->UpdateTensorMap(in, infer_tensor);
      context_->SetPermuteVector(in, trans_pv);
    }

    auto pad_type = TranslatePadType(op_->impl()->node()->nn_param.conv2d.pad_type);
    std::array<uint32_t, 2> ksize = {
      op_->impl()->node()->nn_param.conv2d.ksize[0],
          op_->impl()->node()->nn_param.conv2d.ksize[1]
    };
    std::array<uint32_t, 2> stride = {
      op_->impl()->node()->nn_param.conv2d.stride[0],
      op_->impl()->node()->nn_param.conv2d.stride[1]
    };
    std::array<uint32_t, 2> dilation = {
      op_->impl()->node()->nn_param.conv2d.dilation[0],
      op_->impl()->node()->nn_param.conv2d.dilation[1]
    };
    std::array<uint32_t, 4> pad = {
      op_->impl()->node()->nn_param.conv2d.pad[0],
      op_->impl()->node()->nn_param.conv2d.pad[1],
      op_->impl()->node()->nn_param.conv2d.pad[2],
      op_->impl()->node()->nn_param.conv2d.pad[3]
    };
    int32_t multiplier = op_->impl()->node()->nn_param.conv2d.multiplier;
    int32_t out_channels = op_->impl()->node()->nn_param.conv2d.weights;
    auto conv2d = context_->infer_graph_->CreateOperation<vx::ops::Conv2d>(
        out_channels, pad_type, ksize, stride, dilation, pad, multiplier,
        vx::DataLayout::WHCN, vx::DataLayout::WHIcOc);
    auto otensor_infer = CreateOutputsTensor(required_pv);
    for (const auto& i_src : input_tensors) {
      (*conv2d).BindInput(context_->GetMapedTensor(i_src));
    }
    (*conv2d).BindOutput(otensor_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif
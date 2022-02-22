/****************************************************************************
*
*    Copyright (c) 2022 Vivante Corporation
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
#include "tim/vx/ops.h"

#include "op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

class DenseImpl : public OpImpl {
 public:
  DenseImpl(Graph* graph, int input_cnt, int output_cnt, uint32_t axis,
            uint32_t weights, DataLayout layout = DataLayout::ANY)
      : OpImpl(graph, -1, input_cnt, output_cnt, layout),
        axis_(axis),
        weights_(weights) {
    FC_op_ =
        graph->CreateOperation<tim::vx::ops::FullyConnected>(axis, weights);
  }

  ~DenseImpl() {}

  DenseImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;
    if (this->input_tensor_index == 1) {
      auto input_tensor = in_tensors_[0];
      auto weight_tensor = in_tensors_[1];

      if (input_tensor->GetShape().size() > 2 ||
          (input_tensor->GetShape().size() == 2 &&
           input_tensor->GetShape()[0] != weight_tensor->GetShape()[0])) {
        uint32_t input_size = weight_tensor->GetShape()[0];
        uint32_t total_input_size = 1;
        for (uint8_t i = 0; i < input_tensor->GetShape().size(); i++) {
          total_input_size *= input_tensor->GetShape()[i];
        }
        uint32_t input_batch = total_input_size / input_size;
        tim::vx::TensorSpec reshape_spec(tim::vx::DataType::FLOAT32, {0, 0},
                                         tim::vx::TensorAttribute::TRANSIENT);
        auto reshape_output = graph_->CreateTensor(reshape_spec);
        std::vector<uint32_t> new_shape{input_size, input_batch};
        auto reshape_op =
            graph_->CreateOperation<tim::vx::ops::Reshape>(new_shape);
        (*reshape_op).BindInput(in_tensors_[0]);
        (*reshape_op).BindOutput(reshape_output);
        in_tensors_[0] = reshape_output;
      }
      FC_op_->BindInput(in_tensors_[0]);
      FC_op_->BindInput(in_tensors_[1]);
    }
    if (this->input_tensor_index == 2) {
      FC_op_->BindInput(in_tensors_[input_tensor_index]);
    }
    input_tensor_index++;
    return *this;
  }

  DenseImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;
    if (tensor->GetShape().size() > 2) {
      tim::vx::TensorSpec fc_spec(tim::vx::DataType::FLOAT32, {0, 0},
                                  tim::vx::TensorAttribute::TRANSIENT);
      auto fc_out = graph_->CreateTensor(fc_spec);
      FC_op_->BindOutput(fc_out);
      auto reshape_op =
          graph_->CreateOperation<tim::vx::ops::Reshape>(tensor->GetShape());
      (*reshape_op).BindInput(fc_out);
      (*reshape_op).BindOutput(tensor);
    } else {
      FC_op_->BindOutput(tensor);
    }
    return *this;
  }

  vsi_nn_node_t* node() override { return nullptr; }

  std::vector<std::shared_ptr<Tensor>> InputsTensor() override {
    return inputs_tensor_;
  }
  std::vector<std::shared_ptr<Tensor>> OutputsTensor() override {
    return outputs_tensor_;
  }

  uint32_t axis_;
  uint32_t weights_;

 private:
  std::shared_ptr<tim::vx::Operation> FC_op_;
  std::array<std::shared_ptr<tim::vx::Tensor>, 3> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, 1> out_tensors_;
};

Dense::Dense(Graph* graph, uint32_t axis) : Dense(graph, axis, 0) {}

Dense::Dense(Graph* graph, uint32_t axis, uint32_t weights) {
  impl_ =
      std::make_unique<DenseImpl>(graph, 0, 0, axis, weights, DataLayout::ANY);
}

std::shared_ptr<Operation> Dense::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Dense>(
      dynamic_cast<DenseImpl*>(this->impl_.get())->axis_,
      dynamic_cast<DenseImpl*>(this->impl_.get())->weights_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim

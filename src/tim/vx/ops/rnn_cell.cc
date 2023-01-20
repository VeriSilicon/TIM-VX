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
#include "tim/vx/ops.h"
#include "vsi_nn_pub.h"
#include "op_impl.h"

#include <array>
namespace tim {
namespace vx {
namespace ops {

class RNNCellImpl : public OpImpl {
 public:
  enum {
    // signature
    FULLY_CONNECTED_0_IN = 0,
    FULLY_CONNECTED_0_WEIGHT = 1,
    FULLY_CONNECTED_0_BIAS = 2,
    FULLY_CONNECTED_1_WEIGHT = 3,
    FULLY_CONNECTED_1_STATE_IN = 4,

    INPUT_CNT,

    OUT = 0,
    STATE_OUT,
    OUT_CNT,
    // signature end
  };

  RNNCellImpl(Graph* graph, int input_cnt, int output_cnt,
              DataLayout layout = DataLayout::ANY)
      : OpImpl(graph, -1, input_cnt, output_cnt, layout) {
    fc0_ = graph->CreateOperation<tim::vx::ops::FullyConnected>(0, 4);
    fc1_ = graph->CreateOperation<tim::vx::ops::FullyConnected>(0, 4);
    add_ = graph->CreateOperation<tim::vx::ops::Add>();
    tanh_ = graph->CreateOperation<tim::vx::ops::Tanh>();
    data_convert_ = graph->CreateOperation<tim::vx::ops::DataConvert>();
  }

  ~RNNCellImpl() {}

  RNNCellImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;

    if (this->input_tensor_index == INPUT_CNT - 1) {
      // Get all input tensor
      tim::vx::ShapeType shape = {0, 0};
      tim::vx::TensorSpec FC0_spec(tim::vx::DataType::FLOAT32, shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec FC1_spec(tim::vx::DataType::FLOAT32, shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec add_spec(tim::vx::DataType::FLOAT32, shape,
                                   tim::vx::TensorAttribute::TRANSIENT);

      auto FC0_tensor = graph_->CreateTensor(FC0_spec);
      auto FC1_tensor = graph_->CreateTensor(FC1_spec);
      auto add_tensor = graph_->CreateTensor(add_spec);

      fc0_->BindInput(in_tensors_[FULLY_CONNECTED_0_IN]);
      fc0_->BindInput(in_tensors_[FULLY_CONNECTED_0_WEIGHT]);
      fc0_->BindInput(in_tensors_[FULLY_CONNECTED_0_BIAS]);
      fc0_->BindOutput(FC0_tensor);

      fc1_->BindInput(in_tensors_[FULLY_CONNECTED_1_WEIGHT]);
      fc1_->BindInput(in_tensors_[FULLY_CONNECTED_1_STATE_IN]);
      fc1_->BindOutput(FC1_tensor);

      add_->BindInput(FC0_tensor);
      add_->BindInput(FC1_tensor);
      add_->BindOutput(add_tensor);

      tanh_->BindInput(add_tensor);
    }
    this->input_tensor_index++;
    return *this;
  }

  RNNCellImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;
    if (this->output_tensor_index == OUT_CNT - 1) {
      tanh_->BindOutput(out_tensors_[OUT]);
      data_convert_->BindInput(out_tensors_[OUT]);
      data_convert_->BindOutput(out_tensors_[STATE_OUT]);
    }
    this->output_tensor_index++;
    return *this;
  }

  vsi_nn_node_t* node() override { return nullptr; }

  std::vector<std::shared_ptr<Tensor>> InputsTensor() override {
    return inputs_tensor_;
  }
  std::vector<std::shared_ptr<Tensor>> OutputsTensor() override {
    return outputs_tensor_;
  }

 private:
  std::shared_ptr<tim::vx::Operation> fc0_;
  std::shared_ptr<tim::vx::Operation> fc1_;
  std::shared_ptr<tim::vx::Operation> add_;
  std::shared_ptr<tim::vx::Operation> tanh_;
  std::shared_ptr<tim::vx::Operation> data_convert_;

  std::array<std::shared_ptr<tim::vx::Tensor>, INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, OUT_CNT> out_tensors_;
};

RNNCell::RNNCell(Graph* graph, ActivationType activation)
    : activation_(activation) {
  impl_ = std::make_unique<RNNCellImpl>(graph, 0, 0, DataLayout::ANY);
}

std::shared_ptr<Operation> RNNCell::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<RNNCell>(this->activation_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim

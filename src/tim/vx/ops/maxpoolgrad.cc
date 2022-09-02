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
#ifdef VSI_FEAT_OP_MAXPOOLWITHARGMAX
#include "tim/vx/ops.h"
#include "vsi_nn_pub.h"
#include "op_impl.h"

#include <array>
namespace tim {
namespace vx {
namespace ops {

class MaxpoolGradImpl : public OpImpl {
 public:
  enum {
    TENSOR_BEFORE_POOL = 0,
    UPDATES_TENSOR,
    INPUT_CNT,
    OUT_CNT = 1,
  };
  MaxpoolGradImpl(Graph* graph, PadType padding,
                  const std::array<uint32_t, 2>& ksize,
                  const std::array<uint32_t, 2>& stride,
                  int input_cnt, int output_cnt,
                  RoundType round_type,
                  DataLayout layout)
    : OpImpl(graph, -1, input_cnt, output_cnt, layout),
      padding_(padding),
      ksize_(ksize),
      stride_(stride),
      round_type_(round_type) {
        maxpoolwithargmax2_ = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
          padding_, ksize_, stride_, round_type_, layout_);
  }
  ~MaxpoolGradImpl() {}



  MaxpoolGradImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;
    if (this->input_tensor_index == INPUT_CNT - 1) {
      tim::vx::ShapeType in_shape = in_tensors_[TENSOR_BEFORE_POOL]->GetShape();
      tim::vx::ShapeType updates_shape = in_tensors_[UPDATES_TENSOR]->GetShape();
      tim::vx::ShapeType idx_flattened_shape({CalFlattenedShape(updates_shape)});
      tim::vx::ShapeType out_flattened_shape({CalFlattenedShape(in_shape)});

      tim::vx::TensorSpec pool_out_spec_indices(tim::vx::DataType::INT32,
                              updates_shape, tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec pool_out_spec_values(tim::vx::DataType::FLOAT32,
                              updates_shape, tim::vx::TensorAttribute::OUTPUT);
      tim::vx::TensorSpec idx_flattened_spec(tim::vx::DataType::INT32,
                              idx_flattened_shape, tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec upd_flattened_spec(tim::vx::DataType::FLOAT32,
                              idx_flattened_shape, tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec out_flattened_spec(tim::vx::DataType::FLOAT32,
                              out_flattened_shape, tim::vx::TensorAttribute::TRANSIENT);
      
      auto pool_out_indices_tensor = graph_->CreateTensor(pool_out_spec_indices);
      auto pool_out_values_tensor = graph_->CreateTensor(pool_out_spec_values);
      auto idx_flattened_tensor = graph_->CreateTensor(idx_flattened_spec);
      auto upd_flattened_tensor = graph_->CreateTensor(upd_flattened_spec);
      auto out_flattened_tensor = graph_->CreateTensor(out_flattened_spec);

      (*maxpoolwithargmax2_).BindInput(in_tensors_[TENSOR_BEFORE_POOL])
        .BindOutputs({pool_out_values_tensor, pool_out_indices_tensor});
      
      flatten_idx = graph_->CreateOperation<tim::vx::ops::Reshape>(idx_flattened_shape);
      (*flatten_idx).BindInput(pool_out_indices_tensor).BindOutput(idx_flattened_tensor);

      flatten_upd = graph_->CreateOperation<tim::vx::ops::Reshape>(idx_flattened_shape);
      (*flatten_upd).BindInput(in_tensors_[UPDATES_TENSOR]).BindOutput(upd_flattened_tensor);

      scatternd_ = graph_->CreateOperation<tim::vx::ops::ScatterND>(out_flattened_shape);
      (*scatternd_).BindInputs({idx_flattened_tensor, upd_flattened_tensor}).BindOutput(out_flattened_tensor);

      reshape_like_input_ = graph_->CreateOperation<tim::vx::ops::Reshape>(in_shape);
      (*reshape_like_input_).BindInput(out_flattened_tensor);
    }
    this->input_tensor_index++;
    return *this;
  }

  MaxpoolGradImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;
    (*reshape_like_input_).BindOutput(tensor);
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
  const PadType padding_;
  const std::array<uint32_t, 2> ksize_;
  const std::array<uint32_t, 2> stride_;
  const RoundType round_type_;

  std::shared_ptr<tim::vx::Operation> maxpoolwithargmax2_;
  std::shared_ptr<tim::vx::Operation> flatten_idx;
  std::shared_ptr<tim::vx::Operation> flatten_upd;
  std::shared_ptr<tim::vx::Operation> scatternd_;
  std::shared_ptr<tim::vx::Operation> reshape_like_input_;
  std::array<std::shared_ptr<tim::vx::Tensor>, INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, OUT_CNT> out_tensors_;
  uint32_t CalFlattenedShape(const tim::vx::ShapeType& shape) {
    uint32_t out = 1;
    for(auto& x: shape) {
      out *= x;
    }
    return out;
  }
};

MaxpoolGrad::MaxpoolGrad(Graph* graph, PadType padding,
                         const std::array<uint32_t, 2>& ksize,
                         const std::array<uint32_t, 2>& stride,
                         RoundType round_type,
                         DataLayout layout)
  : padding_(padding),
    ksize_(ksize),
    stride_(stride),
    round_type_(round_type) {
  impl_ = std::make_unique<MaxpoolGradImpl>(graph, padding, ksize, stride, 0, 0, round_type, layout);
}

std::shared_ptr<Operation> MaxpoolGrad::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<MaxpoolGrad>(
      this->padding_, this->ksize_, this->stride_, this->round_type_,
      this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif //(VSI_FEAT_OP_MAXPOOLWITHARGMAX)
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

#ifdef VSI_FEAT_OP_MAXPOOLWITHARGMAX

namespace tim {
namespace vx {
namespace ops {

class MaxpoolGradImpl : public OpImpl {
 public:
  enum {
    POOL_INPUT_TENSOR = 0,
    GRADIENT_TENSOR = 1,
    INPUT_CNT = 2,
    UPDATED_TENSOR = 0,
    OUTPUT_CNT = 1,
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
        maxpoolwithargmax2_ = 
          graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
            padding_, ksize_, stride_, round_type_, layout_);
  }
  ~MaxpoolGradImpl() {}

  MaxpoolGradImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;
    if (this->input_tensor_index == INPUT_CNT - 1) {
      tim::vx::ShapeType in_shape = in_tensors_[POOL_INPUT_TENSOR]->GetShape();
      tim::vx::ShapeType grad_shape = in_tensors_[GRADIENT_TENSOR]->GetShape();
      tim::vx::ShapeType idx_flattened_shape({CalFlattenedShape(grad_shape)});
      tim::vx::ShapeType out_flattened_shape({CalFlattenedShape(in_shape)});

      auto in_type = in_tensors_[POOL_INPUT_TENSOR]->GetDataType();
      auto in_quant = in_tensors_[POOL_INPUT_TENSOR]->GetQuantization();
      if (in_quant.Type() != tim::vx::QuantType::NONE) {
        VSILOGW("MaxPoolGrad deal with quantization tensor not validate yet!");
      }
      tim::vx::TensorSpec pool_out_spec_values(in_type,
          grad_shape, tim::vx::TensorAttribute::TRANSIENT, in_quant);
      tim::vx::TensorSpec pool_out_spec_indices(tim::vx::DataType::INT32,
          grad_shape, tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec idx_flattened_spec(tim::vx::DataType::INT32,
          idx_flattened_shape,tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec upd_flattened_spec(in_type,
          idx_flattened_shape, tim::vx::TensorAttribute::TRANSIENT, in_quant);
      tim::vx::TensorSpec out_flattened_spec(in_type,
          out_flattened_shape, tim::vx::TensorAttribute::TRANSIENT, in_quant);

      auto pool_out_values_tensor = graph_->CreateTensor(pool_out_spec_values);
      auto pool_out_indices_tensor =
          graph_->CreateTensor(pool_out_spec_indices);
      auto idx_flattened_tensor = graph_->CreateTensor(idx_flattened_spec);
      auto upd_flattened_tensor = graph_->CreateTensor(upd_flattened_spec);
      auto out_flattened_tensor = graph_->CreateTensor(out_flattened_spec);

      (*maxpoolwithargmax2_).BindInput(in_tensors_[POOL_INPUT_TENSOR])
          .BindOutputs({pool_out_values_tensor, pool_out_indices_tensor});

      // eliminate pool out of maxpoolwithargmax begin
      tim::vx::TensorSpec sliced_spec(in_type,
          {1, 1, 1, 1}, tim::vx::TensorAttribute::TRANSIENT, in_quant);
      auto sliced_tensor = graph_->CreateTensor(sliced_spec);
      auto one_zero_tensor = graph_->CreateTensor(sliced_spec);
      auto grad_tensor = graph_->CreateTensor(pool_out_spec_values);

      std::vector<int32_t> start = {0, 0, 0, 0};
      std::vector<int32_t> length = {1, 1, 1, 1};
      auto slice_one =
          graph_->CreateOperation<tim::vx::ops::Slice>(0, start, length);
      (*slice_one).BindInput(pool_out_values_tensor).BindOutput(sliced_tensor);

      auto self_sub = graph_->CreateOperation<tim::vx::ops::Sub>();
      (*self_sub).BindInputs({sliced_tensor, sliced_tensor})
                 .BindOutput(one_zero_tensor);

      auto add_zeros = graph_->CreateOperation<tim::vx::ops::Add>();
      (*add_zeros).BindInputs({one_zero_tensor, in_tensors_[GRADIENT_TENSOR]})
                  .BindOutput(grad_tensor);
      // eliminate pool out of maxpoolwithargmax end

      auto flatten_idx =
          graph_->CreateOperation<tim::vx::ops::Reshape>(idx_flattened_shape);
      (*flatten_idx).BindInput(pool_out_indices_tensor)
                    .BindOutput(idx_flattened_tensor);

      auto flatten_upd =
          graph_->CreateOperation<tim::vx::ops::Reshape>(idx_flattened_shape);
      (*flatten_upd).BindInput(grad_tensor).BindOutput(upd_flattened_tensor);

      auto scatternd =
          graph_->CreateOperation<tim::vx::ops::ScatterND>(out_flattened_shape);
      (*scatternd).BindInputs({idx_flattened_tensor, upd_flattened_tensor})
                  .BindOutput(out_flattened_tensor);

      reshape_like_input_ =
          graph_->CreateOperation<tim::vx::ops::Reshape>(in_shape);
      (*reshape_like_input_).BindInput(out_flattened_tensor);

    }
    this->input_tensor_index++;
    return *this;
  }

  MaxpoolGradImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;
    if (this->output_tensor_index == OUTPUT_CNT - 1) {
      (*reshape_like_input_).BindOutput(out_tensors_[UPDATED_TENSOR]);
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
  const PadType padding_;
  const std::array<uint32_t, 2> ksize_;
  const std::array<uint32_t, 2> stride_;
  const RoundType round_type_;

  std::shared_ptr<tim::vx::Operation> maxpoolwithargmax2_;
  std::shared_ptr<tim::vx::Operation> reshape_like_input_;
  std::array<std::shared_ptr<tim::vx::Tensor>, INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, OUTPUT_CNT> out_tensors_;
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
  impl_ = std::make_unique<MaxpoolGradImpl>(graph, padding, ksize, stride, 2, 1, round_type, layout);
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

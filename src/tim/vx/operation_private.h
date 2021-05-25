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
#ifndef TIM_VX_OPERATION_PRIVATE_H_
#define TIM_VX_OPERATION_PRIVATE_H_
#include "graph_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
class OperationImpl {
 public:
  // OperationImpl(Graph* graph, uint32_t operation_id, int input_cnt = 0,
  //               int output_cnt = 0);
  OperationImpl(Graph* graph, uint32_t operation_id, int input_cnt = 0,
                int output_cnt = 0, DataLayout layout = DataLayout::ANY);
  ~OperationImpl() {}

  OperationImpl& BindInput(const std::shared_ptr<Tensor>& tensor);
  OperationImpl& BindOutput(const std::shared_ptr<Tensor>& tensor);
  OperationImpl& SetRoundingPolicy(
      OverflowPolicy overflow_policy = OverflowPolicy::SATURATE,
      RoundingPolicy rounding_policy = RoundingPolicy::RTNE,
      RoundType down_scale_size_rounding = RoundType::FLOOR,
      uint32_t accumulator_bits = 0);

  vsi_nn_node_t* node() { return this->node_; }

  std::vector<std::shared_ptr<Tensor>> InputsTensor() { return inputs_tensor_; }
  std::vector<std::shared_ptr<Tensor>> OutputsTensor() {
    return outputs_tensor_;
  }

  GraphImpl* graph_;
  uint32_t operation_id_{0};
  int32_t input_cnt_{0};
  int32_t output_cnt_{0};
  DataLayout layout_{DataLayout::ANY};
  vsi_nn_node_t* node_{nullptr};
  int32_t input_tensor_index{0};
  int32_t output_tensor_index{0};
  std::vector<std::shared_ptr<Tensor>> inputs_tensor_;
  std::vector<std::shared_ptr<Tensor>> outputs_tensor_;

};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPERATION_PRIVATE_H_ */

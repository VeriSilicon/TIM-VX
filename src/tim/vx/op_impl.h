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

#ifndef TIM_VX_OP_IMPL_H_
#define TIM_VX_OP_IMPL_H_
#include "graph_private.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"

namespace tim {
namespace vx {

class OpImpl {
 public:
  OpImpl(Graph* graph, int32_t kind, int input_cnt, int output_cnt,
         DataLayout layout);
  OpImpl(Graph* graph, DataLayout layout);

  virtual ~OpImpl() = default;
  virtual OpImpl& BindInput(const std::shared_ptr<Tensor>& tensor) = 0;
  virtual OpImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) = 0;
  virtual std::vector<std::shared_ptr<Tensor>> InputsTensor() = 0;
  virtual std::vector<std::shared_ptr<Tensor>> OutputsTensor() = 0;
  virtual vsi_nn_node_t* node() = 0;
  virtual void SetRoundingPolicy(
      OverflowPolicy overflow_policy = OverflowPolicy::SATURATE,
      RoundingPolicy rounding_policy = RoundingPolicy::RTNE,
      RoundType down_scale_size_rounding = RoundType::FLOOR,
      uint32_t accumulator_bits = 0);

  GraphImpl* graph_{nullptr};
  int32_t kind_{0};
  int32_t input_cnt_{0};
  int32_t output_cnt_{0};
  DataLayout layout_{DataLayout::ANY};
  int32_t input_tensor_index{0};
  int32_t output_tensor_index{0};
  std::vector<std::shared_ptr<Tensor>> inputs_tensor_;
  std::vector<std::shared_ptr<Tensor>> outputs_tensor_;
};

}  // namespace vx
}  // namespace tim

#endif
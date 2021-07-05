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
#ifndef TIM_VX_OPERATION_H_
#define TIM_VX_OPERATION_H_

#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

namespace tim {
namespace vx {

class OperationImpl;

class Operation {
 public:
  Operation(Graph* graph, uint32_t operation_id,
            int input_cnt = 0, int ouput_cnt = 0, DataLayout layout = DataLayout::ANY);
  virtual ~Operation();
  virtual std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const = 0;
  Operation& BindInput(const std::shared_ptr<Tensor>& tensor);
  Operation& BindOutput(const std::shared_ptr<Tensor>& tensor);
  Operation& BindInputs(const std::vector<std::shared_ptr<Tensor>>& tensors);
  Operation& BindOutputs(const std::vector<std::shared_ptr<Tensor>>& tensors);
  Operation& SetRoundingPolicy(
      OverflowPolicy overflow_policy = OverflowPolicy::SATURATE,
      RoundingPolicy rounding_policy = RoundingPolicy::RTNE,
      RoundType down_scale_size_rounding = RoundType::FLOOR,
      uint32_t accumulator_bits = 0);
  std::unique_ptr<OperationImpl>& impl();

 protected:
  std::unique_ptr<OperationImpl> impl_;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPERATION_H_ */
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
#ifndef TIM_VX_GRAPH_H_
#define TIM_VX_GRAPH_H_

#include <memory>
#include <vector>

namespace tim {
namespace vx {

class Tensor;
class TensorSpec;

class Operation;

class Graph {
 public:
  virtual ~Graph() {}

  /// Create a tensor with given `TensorSpec`
  virtual std::shared_ptr<Tensor> CreateTensor(const TensorSpec& spec,
                                               const void* data = nullptr) = 0;

  /// Create a placeholder tensor for optional inputs of operations
  virtual std::shared_ptr<Tensor> CreateTensorPlaceHolder() = 0;

  /// Freeze graph
  virtual bool Compile() = 0;

  /// Compile to BinaryGraph
  virtual bool CompileToBinary(void* buf, size_t* size) = 0;

  virtual bool Run() = 0;

  template <typename OpType, typename... Params>
  std::shared_ptr<OpType> CreateOperation(Params... parameters) {
    auto op = std::make_shared<OpType>(this, parameters...);
    opVector.push_back(op);
    return op;
  }

private:
  std::vector<std::shared_ptr<tim::vx::Operation>> opVector;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_GRAPH_H_ */
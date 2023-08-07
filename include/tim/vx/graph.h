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
#ifndef TIM_VX_GRAPH_H_
#define TIM_VX_GRAPH_H_

#ifdef BUILD_WITH_BAZEL
#include "vsi_feat_ops_def.h"
#endif
#ifdef ENABLE_TENSOR_CACHE
#include <openssl/evp.h>
#include <string>
#endif
#include <memory>
#include <vector>
#include <map>
namespace tim {
namespace vx {
#ifdef ENABLE_TENSOR_CACHE
const std::string calculateMd5Secret32(const std::string& src);
#endif
class Tensor;
struct TensorSpec;
struct DmaBufferDesc;
class Operation;
class CompileOption;

class Graph {
 public:
  virtual ~Graph() {}

  /// Attach CompileOption
  virtual void SetCompileOption(const CompileOption&) = 0;

  /// Create a tensor with given `TensorSpec`
  virtual std::shared_ptr<Tensor> CreateTensor(const TensorSpec& spec,
                                               const void* data = nullptr) = 0;

  virtual std::shared_ptr<Tensor> CreateTensor(const TensorSpec& spec,
                                               const DmaBufferDesc& dmafd) = 0;

  /// Create a tensor with given `TensorSpec`.
  /// spec.attr_ must be TensorAttribute::Input or Output
  virtual std::shared_ptr<Tensor> CreateIOTensor(const TensorSpec& spec,
                                                 void* data = nullptr) = 0;

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
    op_vector_.push_back(op);
    return op;
  }

  virtual const std::vector<std::shared_ptr<Tensor>> InputsTensor() const = 0;
  virtual const std::vector<std::shared_ptr<Tensor>> OutputsTensor() const = 0;

  virtual void UpdateTensorConsumersMap(const std::shared_ptr<Tensor>& tensor,
                                        const Operation* op) = 0;
  virtual void RenewTensorConsumersMap(
      const std::shared_ptr<Tensor>& org_tensor,
      const std::shared_ptr<Tensor>& dst_tensor, const Operation* op) = 0;

  virtual void UpdateTensorProducerMap(const std::shared_ptr<Tensor>& tensor,
                                       const Operation* op) = 0;

  virtual const std::vector<std::shared_ptr<Operation>> GetConsumersOp(
      std::shared_ptr<Tensor> tensor) const = 0;

  virtual std::shared_ptr<Operation> GetProducerOp(
      std::shared_ptr<Tensor> tensor) = 0;

  virtual void PrintGraph() const = 0;

  const std::vector<std::shared_ptr<Tensor>> GetConstantInputs() const;
  virtual std::vector<std::shared_ptr<Operation>>& OpVector() = 0;
  virtual std::map<std::shared_ptr<Tensor>,
                   std::vector<std::shared_ptr<Operation>>>&
  TensorConsumer() = 0;
  virtual std::map<std::shared_ptr<Tensor>, std::shared_ptr<Operation>>&
  TensorProducer() = 0;

 protected:
  std::vector<std::shared_ptr<tim::vx::Operation>> op_vector_;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_GRAPH_H_ */
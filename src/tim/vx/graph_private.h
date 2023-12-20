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
#ifndef TIM_VX_GRAPH_PRIVATE_H_
#define TIM_VX_GRAPH_PRIVATE_H_
#include "tim/vx/graph.h"

#include <vector>
#include <string>
#include <mutex>
#include <utility>
#include <map>

#include "tim/vx/tensor.h"
#include "tim/vx/compile_option.h"
#include "context_private.h"

#include "vsi_nn_pub.h"

namespace tim {
namespace vx {

class GraphImpl : public Graph {
 public:
  GraphImpl(ContextImpl* context,
            const CompileOption& options = CompileOption::DefaultOptions);
  ~GraphImpl();
#ifdef ENABLE_TENSOR_CACHE
  std::shared_ptr<Tensor> GetTensorFromCache(const TensorSpec& spec,
                                             const void* data);
  const std::string CalculateCacheKey(const TensorSpec& spec, const void* data);
  std::map<std::string, std::shared_ptr<tim::vx::Tensor>>& GetTensorCacheMap();
#endif

  void SetCompileOption(const CompileOption& new_option) override;

  /// Return the low-level graph object
  vsi_nn_graph_t* graph();
  void AddInput(vsi_nn_tensor_id_t id);
  void AddOutput(vsi_nn_tensor_id_t id);

  void AddInput(const std::shared_ptr<Tensor>& tensor);
  void AddOutput(const std::shared_ptr<Tensor>& tensor);

  const std::vector<std::shared_ptr<Tensor>> InputsTensor() const override;
  const std::vector<std::shared_ptr<Tensor>> OutputsTensor() const override;
  std::vector<std::shared_ptr<Operation>>& OpVector() override;
  std::map<std::shared_ptr<Tensor>, std::vector<std::shared_ptr<Operation>>>&
  TensorConsumer() override;
  std::map<std::shared_ptr<Tensor>, std::shared_ptr<Operation>>&
  TensorProducer() override;
  void UpdateTensorConsumersMap(const std::shared_ptr<Tensor>& tensor,
                                const Operation* op) override;
  void RenewTensorConsumersMap(const std::shared_ptr<Tensor>& org_tensor,
                               const std::shared_ptr<Tensor>& dst_tensor,
                               const Operation* op) override;
  void UpdateTensorProducerMap(const std::shared_ptr<Tensor>& tensor,
                               const Operation* op) override;
  const std::vector<std::shared_ptr<Operation>> GetConsumersOp(
      std::shared_ptr<Tensor> tensor) const override;
  std::shared_ptr<Operation> GetProducerOp(
      std::shared_ptr<Tensor> tensor) override;

  void PrintGraph() const override;

  std::shared_ptr<Tensor> CreateTensor(const TensorSpec& spec,
                                       const void* data = nullptr) override;
  std::shared_ptr<Tensor> CreateTensor(const TensorSpec& spec,
                                       const DmaBufferDesc& dmafd) override;
  std::shared_ptr<Tensor> CreateIOTensor(const TensorSpec& spec,
                                         void* data = nullptr) override;
  std::shared_ptr<Tensor> CreateTensorPlaceHolder() override;

  bool Compile() override;
  bool CompileToBinary(void* buf, size_t* size) override;
  bool Run() override;
  void ProduceInput() { not_consumed_input_cnt_++; }
  void ProduceOutput() { not_consumed_output_cnt_++; }
  void ConsumeInput() { not_consumed_input_cnt_--; }
  void ConsumeOutput() { not_consumed_output_cnt_--; }

 protected:
  ContextImpl* context_;
  vsi_nn_graph_t* graph_;
  std::shared_ptr<Tensor> tensor_placeholder_;
  std::once_flag setio_once_;
  std::once_flag setup_once_;
  std::once_flag verify_graph_once_;
  std::vector<vsi_nn_tensor_id_t> inputs_;
  std::vector<vsi_nn_tensor_id_t> outputs_;
  std::vector<std::shared_ptr<Tensor>> inputs_tensor_;
  int32_t not_consumed_input_cnt_;
  std::vector<std::shared_ptr<Tensor>> outputs_tensor_;
  int32_t not_consumed_output_cnt_;
  std::map<std::shared_ptr<Tensor>, std::vector<std::shared_ptr<Operation>>>
      tensor_consumers_;
  std::map<std::shared_ptr<Tensor>, std::shared_ptr<Operation>>
      tensor_producer_;
#ifdef ENABLE_TENSOR_CACHE
  std::map<std::string, std::shared_ptr<tim::vx::Tensor>> cached_tensor_;
#endif
  CompileOption options_;

 private:
  /// Setup graph
  bool Setup();
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_GRAPH_PRIVATE_H_ */
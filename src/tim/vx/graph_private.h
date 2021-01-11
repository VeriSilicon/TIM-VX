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
#ifndef TIM_VX_GRAPH_PRIVATE_H_
#define TIM_VX_GRAPH_PRIVATE_H_
#include <vector>

#include "context_private.h"
#include "tim/vx/graph.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {

class GraphImpl : public Graph {
 public:
  GraphImpl(ContextImpl* context);
  ~GraphImpl();

  /// Return the low-level graph object
  vsi_nn_graph_t* graph();

  void AddInput(vsi_nn_tensor_id_t id);
  void AddOutput(vsi_nn_tensor_id_t id);

  /// Implement parents' virtual functions
  std::shared_ptr<Tensor> CreateTensor(const TensorSpec& spec,
                                       const void* data = nullptr);
  std::shared_ptr<Tensor> CreateTensorPlaceHolder();
  bool Compile();
  bool Run();

 protected:
  ContextImpl* context_;
  vsi_nn_graph_t* graph_;
  std::shared_ptr<Tensor> tensor_placeholder_;
  bool compiled_;
  std::vector<vsi_nn_tensor_id_t> inputs_;
  std::vector<vsi_nn_tensor_id_t> outputs_;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_GRAPH_PRIVATE_H_ */
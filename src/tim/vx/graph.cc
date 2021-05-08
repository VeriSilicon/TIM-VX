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
#include "tim/vx/graph.h"

#include <algorithm>

#include "context_private.h"
#include "graph_private.h"
#include "tensor_private.h"
#include "operation_private.h"

#include "tim/vx/context.h"
#include "tim/vx/ops/nbg.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {

GraphImpl::GraphImpl(ContextImpl* context)
    : context_(context),
      graph_(vsi_nn_CreateGraph(context_->context(), 0, 0)),
      tensor_placeholder_(nullptr) {}

GraphImpl::~GraphImpl() { vsi_nn_ReleaseGraph(&graph_); }

vsi_nn_graph_t* GraphImpl::graph() { return graph_; }

void GraphImpl::AddInput(vsi_nn_tensor_id_t id) {
  if (inputs_.end() == std::find(inputs_.begin(), inputs_.end(), id)) {
    inputs_.push_back(id);
  }
}

void GraphImpl::AddOutput(vsi_nn_tensor_id_t id) {
  if (outputs_.end() == std::find(outputs_.begin(), outputs_.end(), id)) {
    outputs_.push_back(id);
  }
}

void GraphImpl::AddInput(const std::shared_ptr<Tensor>& tensor) {
  if (inputs_tensor_.end() ==
      std::find(inputs_tensor_.begin(), inputs_tensor_.end(), tensor)) {
    inputs_tensor_.push_back(tensor);
  }
}

void GraphImpl::AddOutput(const std::shared_ptr<Tensor>& tensor) {
  if (outputs_tensor_.end() ==
      std::find(outputs_tensor_.begin(), outputs_tensor_.end(), tensor)) {
    outputs_tensor_.push_back(tensor);
  }
}

const std::vector<std::shared_ptr<Tensor>> GraphImpl::InputsTensor() const {
  return inputs_tensor_;
}

const std::vector<std::shared_ptr<Tensor>> GraphImpl::OutputsTensor() const {
  return outputs_tensor_;
}

void GraphImpl::UpdateTensorConsumersMap(const std::shared_ptr<Tensor>& tensor,
                                         const Operation* op) {
  for (const auto& added_op : op_vector_) {
    if (added_op.get() == op) {
      tensor_consumers_[tensor].push_back(added_op);
    }
  }
}

const std::vector<std::shared_ptr<Operation>> GraphImpl::GetConsumersOp(
    std::shared_ptr<Tensor> tensor) const {
  auto consumers = tensor_consumers_.find(tensor);
  if (tensor_consumers_.end() != consumers) {
    return consumers->second;
  } else {
    VSILOGD("Tensor has no consumers, may be graph output.");
    return {};
  }
}

void GraphImpl::PrintGraph() const { vsi_nn_PrintGraph(this->graph_); }

std::shared_ptr<Tensor> GraphImpl::CreateTensor(const TensorSpec& spec,
                                                const void* data) {
  return std::make_shared<TensorImpl>(this, spec, data);
}

std::shared_ptr<Tensor> GraphImpl::CreateTensorPlaceHolder() {
  if (!tensor_placeholder_) {
    tensor_placeholder_ = std::make_shared<TensorPlaceholder>(this);
  }

  return tensor_placeholder_;
}

bool GraphImpl::Compile() {
  bool status = true;

  auto major = vsi_nn_GetVersionMajor();
  auto minor = vsi_nn_GetVersionMinor();
  auto patch = vsi_nn_GetVersionPatch();

  vsi_nn_SetGraphVersion(graph_,major,minor,patch);

  std::call_once(setio_once_, [&status, this]() {
    status = (vsi_nn_SetGraphInputs(this->graph_, this->inputs_.data(), this->inputs_.size()) &&
              vsi_nn_SetGraphOutputs(this->graph_, this->outputs_.data(), this->outputs_.size()));
  });

  std::call_once(setup_once_, [&status, this](){
    status = (VSI_SUCCESS == vsi_nn_SetupGraph(this->graph_, true));
  });

  std::call_once(verify_graph_once_, [&status, this]() {
    status = (VSI_SUCCESS == vsi_nn_VerifyGraph(this->graph_));
  });

  return status;
}

bool GraphImpl::CompileToBinary(void* buf, size_t* size) {
  bool status = true;
  std::call_once(setio_once_, [&status, this]() {
    status = (vsi_nn_SetGraphInputs(this->graph_, this->inputs_.data(), this->inputs_.size()) &&
              vsi_nn_SetGraphOutputs(this->graph_,this->outputs_.data(), this->outputs_.size()));
  });

  std::call_once(setup_once_, [&status, this](){
    status = (VSI_SUCCESS == vsi_nn_SetupGraph(this->graph_, true));
  });

  return ((status) && (VSI_SUCCESS == vsi_nn_GenerateNBG(graph_, buf, size)));
}

bool GraphImpl::Run() {
  return ((Compile()) && (VSI_SUCCESS == vsi_nn_RunGraph(graph_)));
}

}  // namespace vx
}  // namespace tim

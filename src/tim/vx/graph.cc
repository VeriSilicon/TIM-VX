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
#include "tim/vx/context.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {

GraphImpl::GraphImpl(ContextImpl* context)
    : context_(context),
      graph_(vsi_nn_CreateGraph(context_->context(), 0, 0)),
      tensor_placeholder_(nullptr),
      compiled_(false) {}

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
  compiled_ = true;

  vsi_nn_SetGraphInputs(graph_, inputs_.data(), inputs_.size());
  vsi_nn_SetGraphOutputs(graph_, outputs_.data(), outputs_.size());

  return (VSI_SUCCESS == vsi_nn_SetupGraph(graph_, true) &&
          VSI_SUCCESS == vsi_nn_VerifyGraph(graph_));
}

bool GraphImpl::Run() {
  if (!compiled_ && !Compile()) {
    return false;
  }
  return (VSI_SUCCESS == vsi_nn_RunGraph(graph_));
}

}  // namespace vx
}  // namespace tim

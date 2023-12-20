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
#include "tim/vx/graph.h"
#include <algorithm>

#ifdef ENABLE_TENSOR_CACHE
#include <openssl/evp.h>
#include <cstring>
#endif

#include "context_private.h"
#include "graph_private.h"
#include "op_impl.h"
#include "tensor_private.h"
#include "tim/vx/context.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/compile_option.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
#ifdef ENABLE_TENSOR_CACHE
#define MD5_SECRET_LEN_16 (16)
#define MD5_BYTE_STRING_LEN (4)
const std::string calculateMd5Secret32(const std::string& src) {
  std::string md5String;
  EVP_MD_CTX* mdctx;
  const EVP_MD* md;
  uint32_t md_len;
  unsigned char md_value[MD5_SECRET_LEN_16] = {0};
  char tmp[MD5_BYTE_STRING_LEN] = {0};

  md = EVP_md5();
  if (md == NULL) {
    printf("Unknown EVP_md5 message.");
  }
  mdctx = EVP_MD_CTX_new();
  if (!EVP_DigestInit_ex(mdctx, md, NULL)) {
    printf("EVP_MD_CTX initialization failed.");
    EVP_MD_CTX_free(mdctx);
  }
  if (!EVP_DigestUpdate(mdctx, src.c_str(), src.size())) {
    printf("EVP_MD_CTX update failed.");
    EVP_MD_CTX_free(mdctx);
  }
  if (!EVP_DigestFinal_ex(mdctx, md_value, &md_len)) {
    printf("EVP_MD_CTX finalization failed.");
    EVP_MD_CTX_free(mdctx);
  }
  EVP_MD_CTX_free(mdctx);

  for (int i = 0; i < MD5_SECRET_LEN_16; ++i) {
    memset(tmp, 0x00, sizeof(tmp));
    snprintf(tmp, sizeof(tmp), "%02X", md_value[i]);
    md5String += tmp;
  }
  return md5String;
}
#endif

const std::vector<std::shared_ptr<Tensor>> Graph::GetConstantInputs() const {
  std::vector<std::shared_ptr<Tensor>> const_inputs;
  for (auto op : op_vector_) {
    auto const_i = op->ConstantInputsTensor();
    const_inputs.insert(const_inputs.end(), const_i.begin(), const_i.end());
  }
  return const_inputs;
}

GraphImpl::GraphImpl(ContextImpl* context, const CompileOption& options)
    : context_(context),
      graph_(vsi_nn_CreateGraph(context_->context(), 0, 0)),
      tensor_placeholder_(nullptr),
      not_consumed_input_cnt_(0),
      not_consumed_output_cnt_(0),
      options_(options) {}

GraphImpl::~GraphImpl() { vsi_nn_ReleaseGraph(&graph_); }

#ifdef ENABLE_TENSOR_CACHE
std::map<std::string, std::shared_ptr<tim::vx::Tensor>>&
GraphImpl::GetTensorCacheMap() {
  return cached_tensor_;
}

const std::string GraphImpl::CalculateCacheKey(const TensorSpec& spec,
                                               const void* data) {
  std::string md5_key;
  uint32_t data_size = 1;
  for (auto it = spec.shape_.begin(); it != spec.shape_.end(); ++it) {
    data_size *= *it;
  }
  switch (spec.datatype_) {
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FLOAT16:
      data_size *= 2;
      break;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      data_size *= 4;
      break;
    case DataType::INT64:
      data_size *= 8;
      break;
    default:
      break;
  }
  if (data_size < 512) {
    md5_key = calculateMd5Secret32(std::string((const char*)data, data_size));
  } else {
    md5_key = calculateMd5Secret32(
        std::string((const char*)data, 512));  //Take first 512 bytes
  }
  return md5_key;
}

std::shared_ptr<Tensor> GraphImpl::GetTensorFromCache(const TensorSpec& spec,
                                                      const void* data) {
  std::shared_ptr<tim::vx::Tensor> tensor;
  std::string md5_key = CalculateCacheKey(spec, data);
  if (GetTensorCacheMap().find(md5_key) != GetTensorCacheMap().end() &&
      GetTensorCacheMap()[md5_key]->GetSpec() == spec) {
    tensor = GetTensorCacheMap()[md5_key];
  } else {
    tensor = std::make_shared<TensorImpl>(this, spec, data);
    GetTensorCacheMap()[md5_key] = tensor;
  }
  return tensor;
}
#endif

void GraphImpl::SetCompileOption(const CompileOption& new_options) {
  options_ = new_options;
}

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

std::vector<std::shared_ptr<Operation>>& GraphImpl::OpVector() {
  return op_vector_;
}

std::map<std::shared_ptr<Tensor>, std::vector<std::shared_ptr<Operation>>>&
GraphImpl::TensorConsumer() {
  return tensor_consumers_;
}

std::map<std::shared_ptr<Tensor>, std::shared_ptr<Operation>>&
GraphImpl::TensorProducer() {
  return tensor_producer_;
}

void GraphImpl::UpdateTensorConsumersMap(const std::shared_ptr<Tensor>& tensor,
                                         const Operation* op) {
  for (const auto& added_op : op_vector_) {
    if (added_op.get() == op) {
      tensor_consumers_[tensor].push_back(added_op);
    }
  }
}

void GraphImpl::RenewTensorConsumersMap(
    const std::shared_ptr<Tensor>& org_tensor,
    const std::shared_ptr<Tensor>& dst_tensor, const Operation* op) {
  auto exist_op = std::find_if(
      op_vector_.begin(), op_vector_.end(),
      [op](std::shared_ptr<Operation> oper) { return oper.get() == op; });
  if (exist_op == op_vector_.end()) {
    return;  //given op cannot be found
  } else {
    auto consumer_to_remove = tensor_consumers_.find(org_tensor);
    if (consumer_to_remove != tensor_consumers_.end())
      tensor_consumers_.erase(consumer_to_remove);
    tensor_consumers_[dst_tensor].push_back(*exist_op);
  }
}

void GraphImpl::UpdateTensorProducerMap(const std::shared_ptr<Tensor>& tensor,
                                        const Operation* op) {
  for (const auto& added_op : op_vector_) {
    if (added_op.get() == op) {
      tensor_producer_[tensor] = added_op;
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

std::shared_ptr<Operation> GraphImpl::GetProducerOp(
    std::shared_ptr<Tensor> tensor) {
  auto producer = tensor_producer_.find(tensor);
  if (tensor_producer_.end() != producer) {
    return producer->second;
  } else {
    VSILOGD("Tensor has no producer, may be graph input.");
    return {};
  }
}

void GraphImpl::PrintGraph() const { vsi_nn_PrintGraph(this->graph_); }

std::shared_ptr<Tensor> GraphImpl::CreateTensor(const TensorSpec& spec,
                                                const void* data) {
#ifdef ENABLE_TENSOR_CACHE
  if (spec.attr_ & TensorAttribute::CONSTANT && data != NULL) {
    return GetTensorFromCache(spec, data);
  }
#endif
  auto tensor = std::make_shared<TensorImpl>(this, spec, data);
  if (spec.attr_ & TensorAttribute::INPUT) {
    this->AddInput(tensor);
    this->AddInput(tensor->GetId());
    this->ProduceInput();
  }
  if (spec.attr_ & TensorAttribute::OUTPUT) {
    this->AddOutput(tensor);
    this->AddOutput(tensor->GetId());
    this->ProduceOutput();
  }
  return tensor;
}

std::shared_ptr<Tensor> GraphImpl::CreateTensor(const TensorSpec& spec,
                                                const DmaBufferDesc& dmafd) {
  auto tensor = std::make_shared<TensorImpl>(this, spec, dmafd);
  if (spec.attr_ & TensorAttribute::INPUT) {
    this->AddInput(tensor);
    this->AddInput(tensor->GetId());
    this->ProduceInput();
  }
  if (spec.attr_ & TensorAttribute::OUTPUT) {
    this->AddOutput(tensor);
    this->AddOutput(tensor->GetId());
    this->ProduceOutput();
  }
  return tensor;
}

std::shared_ptr<Tensor> GraphImpl::CreateIOTensor(const TensorSpec& spec,
                                                  void* data) {
  auto tensor = std::make_shared<TensorImpl>(this, spec, data);
  if (spec.attr_ & TensorAttribute::INPUT) {
    this->AddInput(tensor);
    this->AddInput(tensor->GetId());
    this->ProduceInput();
  }
  if (spec.attr_ & TensorAttribute::OUTPUT) {
    this->AddOutput(tensor);
    this->AddOutput(tensor->GetId());
    this->ProduceOutput();
  }
  return tensor;
}

std::shared_ptr<Tensor> GraphImpl::CreateTensorPlaceHolder() {
  if (!tensor_placeholder_) {
    tensor_placeholder_ = std::make_shared<TensorPlaceholder>(this);
  }

  return tensor_placeholder_;
}

bool GraphImpl::Setup() {
  bool status = true;

  auto major = vsi_nn_GetVersionMajor();
  auto minor = vsi_nn_GetVersionMinor();
  auto patch = vsi_nn_GetVersionPatch();

  vsi_nn_SetGraphVersion(graph_, major, minor, patch);

  bool is_fast_mode = options_.isRelaxMode();
  if (is_fast_mode) {
    VSILOGW(
        "Important notice: float model executed in bfloat16 "
        "mode which will have better performance but lower precesion");
  }
  vsi_nn_SetGraphFastMode(graph_, is_fast_mode);

#if defined(ENABLE_PLATFORM)
  auto id = options_.getDeviceId();
  vxSetGraphAttribute(graph_->g, VX_GRAPH_DEVICE_INDEX_VIV, (void*)(&id),
                      sizeof(id));
#endif

  std::call_once(setio_once_, [&status, this]() {
    status = (vsi_nn_SetGraphInputs(this->graph_, this->inputs_.data(),
                                    this->inputs_.size()) &&
              vsi_nn_SetGraphOutputs(this->graph_, this->outputs_.data(),
                                     this->outputs_.size()));
  });

  std::call_once(setup_once_, [&status, this]() {
    status = (VSI_SUCCESS == vsi_nn_SetupGraph(this->graph_, true));
  });
  return status;
}

bool GraphImpl::Compile() {
  bool status = true;
  if (not_consumed_input_cnt_ > 0) {
    // Tensor can bind to different operations
    VSILOGW(
        "Graph has free input, INPUT tensor may be created but not "
        "consumed.");
  }
  if (not_consumed_output_cnt_ != 0) {
    VSILOGW(
        "Graph has free output, OUTPUT tensor may be created but not "
        "consumed.");
  }
  status = Setup();
  std::call_once(verify_graph_once_, [&status, this]() {
    status = (VSI_SUCCESS == vsi_nn_VerifyGraph(this->graph_));
  });

  return status;
}

bool GraphImpl::CompileToBinary(void* buf, size_t* size) {
  return ((Setup()) && (VSI_SUCCESS == vsi_nn_GenerateNBG(graph_, buf, size)));
}

bool GraphImpl::Run() {
  return ((Compile()) && (VSI_SUCCESS == vsi_nn_RunGraph(graph_)));
}

}  // namespace vx
}  // namespace tim

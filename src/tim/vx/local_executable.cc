#include "tim/vx/LocalExecutable.h"
#include "graph_private.h"
#include "tensor_private.h"
#include "tim/vx/LocalExecutor.h"
#include "local_device_private.h"
#include <chrono>
#include <thread>

namespace tim {
namespace vx {

LocalExecutable::LocalExecutable(std::shared_ptr<IExecutor>& executor, std::vector<char> nb_buf, size_t inputs, size_t outputs) {
  executor_ = executor;
  context_ = executor->Contex();
  nb_graph_ = context_->CreateGraph();
  nb_buf_ = nb_buf;
  nb_node_ = nb_graph_->CreateOperation<tim::vx::ops::NBG>(nb_buf_.data(), inputs, outputs);
}

void LocalExecutable::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  auto local_handle= dynamic_cast<const LocalTensorHandle*> (th.get()); // hack to downcast
  nb_node_->BindInput(local_handle->tensor());
}

void LocalExecutable::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  auto local_handle= dynamic_cast<const LocalTensorHandle*> (th.get()); // hack to downcast
  nb_node_->BindOutput(local_handle->tensor());
}

bool LocalExecutable::Submit(std::shared_ptr<IExecutable> ref, bool after) {
  bool status = false;
  std::shared_ptr<IExecutable> executable = shared_from_this();
  status = executor_.lock()->Submit(executable, ref, after);
  return status;
}

bool LocalExecutable::Trigger(bool async) {
  (void)async;
  bool status = false;
  auto device = executor_.lock()->Device();
  device->Submit(nb_graph_);
  status = device->Trigger();
  device->WaitDeviceIdle();
  return status;
}

std::shared_ptr<ITensorHandle> LocalExecutable::AllocateTensor(const TensorSpec& tensor_spec){
  auto tensor = nb_graph_->CreateTensor(tensor_spec);
  ITensorHandle* tensor_handle = new LocalTensorHandle(tensor);
  std::shared_ptr<ITensorHandle> tensor_handle_sp (tensor_handle);
  return tensor_handle_sp;
}

bool LocalExecutable::NBCompile(){
  return nb_graph_->Compile();
}

}  // namespace vx
}  // namespace tim
#include "tim/vx/LocalExecutor.h"
#include "graph_private.h"

namespace tim {
namespace vx {

  LocalExecutor::LocalExecutor(std::shared_ptr<IDevice> device) {
    device_ = device;
    context_ = tim::vx::Context::Create();
  }

  LocalExecutor::LocalExecutor(std::shared_ptr<IDevice> device, std::shared_ptr<Context> context) {
    device_ = device;
    context_ = context;
  }

  bool LocalExecutor::Submit(const std::shared_ptr<IExecutable> executable, const std::shared_ptr<IExecutable> ref, bool after) {
    bool success = false;
    success = executable->NBCompile();
    if (success == false){
      std::cout<<"Executable NBG compile failed";
      return false;
    }
    if (executable == ref){
      tasks_.push_back(executable);
      return true;
    }
    for (size_t i = 0; i < tasks_.size(); i++){
      if (tasks_[i].lock() == ref){
        if (after == true){
          tasks_.insert(tasks_.begin()+i+1, executable);
          success = true;
          break;
        }
        else{
          tasks_.insert(tasks_.begin()+i, executable);
          success = true;
          break;
        }
      }
    }
    return success;
  }

  bool LocalExecutor::Trigger(bool async) {
    (void)async;
    for (auto task: tasks_){
      task.lock()->Trigger();
    }
    device_->DeviceExit();
    return true;
  }

  std::shared_ptr<IExecutable> LocalExecutor::Compile(const std::shared_ptr<Graph>& graph) {
      GraphImpl* graphimp= dynamic_cast<GraphImpl*> (graph.get()); // hack to downcast
      IDevice::device_id_t id = device_->device_id();
      vxSetGraphAttribute(graphimp->graph()->g, VX_GRAPH_DEVICE_INDEX_VIV, (void*)(&id), sizeof(id));
      size_t bin_size = -1;
      graph->CompileToBinary(nullptr, &bin_size);
      std::vector<char> nb_buf;
      nb_buf.resize(bin_size);
      size_t inputs = graph->InputsTensor().size();
      size_t outputs = graph->OutputsTensor().size();
      graph->CompileToBinary(nb_buf.data(), &bin_size);
      std::shared_ptr<IExecutor> this_sp = shared_from_this();
      IExecutable* executable = new LocalExecutable(this_sp, nb_buf, inputs, outputs);
      std::shared_ptr<IExecutable> executable_sp(executable);
      return executable_sp;
  }

  std::shared_ptr<IDevice> LocalExecutor::Device() {
    return device_;
  }

}  // namespace vx
}  // namespace tim
#include "tim/vx/IExecutableSet.h"
#include "graph_private.h"
#include "tensor_private.h"
#include "tim/vx/context.h"
#include "tim/vx/LocalExecutable.h"
#include "local_device_private.h"
#include "tim/vx/LocalExecutor.h"

namespace tim {
namespace vx {

std::shared_ptr<IExecutable> CreateIExecutableSet(std::vector<std::shared_ptr<IExecutable>> executables){
  IExecutableSet* executable_set = new IExecutableSet(executables);
  std::shared_ptr<IExecutable> executable(executable_set);
  return executable;
}

IExecutableSet::IExecutableSet(std::vector<std::shared_ptr<IExecutable>> executables) {
  executables_ = executables;
  executor_ = executables[0]->Executor();
}

void IExecutableSet::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  (void)th;
}

void IExecutableSet::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  (void)th;
}

bool IExecutableSet::Submit(std::shared_ptr<IExecutable> ref, bool after) {
  bool status = false;
  std::shared_ptr<IExecutable> executable = shared_from_this();
  status = executor_.lock()->Submit(executable, ref, after);
  return status;
}

bool IExecutableSet::Trigger(bool async) {
  (void)async;
  bool status = false;
  auto device = executor_.lock()->Device();
  for ( auto executable : executables_ ){
    device->Submit(executable->NBGraph());
  }
  status = device->Trigger();
  device->WaitDeviceIdle();
  return status;
}

std::shared_ptr<ITensorHandle> IExecutableSet::AllocateTensor(const TensorSpec& tensor_spec){
  std::shared_ptr<ITensorHandle> tensor_handle_sp;
  (void) tensor_spec;
  return tensor_handle_sp;
}

std::vector<std::shared_ptr<IExecutable>> IExecutableSet::Executables(){
  return executables_;
}

bool IExecutableSet::NBCompile(){
  bool status = false;
  for ( auto executable : executables_ ){
    status = executable->NBCompile();
  }
  return status;
}

}  // namespace vx
}  // namespace tim
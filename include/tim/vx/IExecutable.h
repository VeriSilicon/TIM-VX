#ifndef TIM_VX_IEXECUTABLE_H_
#define TIM_VX_IEXECUTABLE_H_
#include <iostream>
#include "IExecutor.h"
#include "tim/vx/context.h"
#include "tim/vx/ops/nbg.h"

namespace tim {
namespace vx {
class ITensorHandle;
class IExecutor;
class IExecutable : public std::enable_shared_from_this<IExecutable>{
 public:
  virtual ~IExecutable(){
    std::cout<<"Destructor IExecutable\n";
    };
  virtual void SetInput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void SetOutput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual bool Submit(std::shared_ptr<IExecutable> ref, bool after=true) = 0;
  virtual bool Trigger(bool async=false) = 0;
  std::shared_ptr<Graph> NBGraph() const {return nb_graph_;}
  virtual bool NBCompile() = 0;
  virtual std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec) = 0;
  virtual std::shared_ptr<IExecutor> Executor() {return executor_.lock();};

 protected:
  std::weak_ptr<IExecutor> executor_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<Graph> nb_graph_;
};

  std::shared_ptr<IExecutable> CreateIExecutableSet(std::vector<std::shared_ptr<IExecutable>> executables);

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_IEXECUTABLE_H_*/
#ifndef TIM_VX_LOCALEXECUTOR_H_
#define TIM_VX_LOCALEXECUTOR_H_
#include <memory>
# include "LocalDevice.h"
# include "IExecutor.h"
# include "LocalExecutable.h"

namespace tim {
namespace vx {

class LocalExecutor : public IExecutor, public std::enable_shared_from_this<LocalExecutor>{
  public:
    LocalExecutor(std::shared_ptr<IDevice> device);
    LocalExecutor(std::shared_ptr<IDevice> device, std::shared_ptr<Context> context);
    ~LocalExecutor(){
      printf("Destructor LocalExecutor: %p\n", this);
      };
    bool Submit(const std::shared_ptr<IExecutable> executable, const std::shared_ptr<IExecutable> ref, bool after=true) override;
    bool Trigger(bool async=false) override;
    std::shared_ptr<IExecutable> Compile(const std::shared_ptr<Graph>& graph/*, const CompileOption& compile_option*/) override;
    std::shared_ptr<IDevice> Device() override;

  protected:
    // std::vector<char> nb_buf_;

};

  // std::shared_ptr<IExecutable> CreateIExecutableSet(std::shared_ptr<LocalExecutor> local_executor, std::vector<std::shared_ptr<IExecutable>> executables);

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_LOCALEXECUTOR_H_*/
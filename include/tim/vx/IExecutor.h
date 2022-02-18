#ifndef TIM_VX_IEXECUTOR_H_
#define TIM_VX_IEXECUTOR_H_
#include <memory>
#include <iostream>
# include "IDevice.h"
# include "IExecutable.h"
#include "tim/vx/context.h"

namespace tim {
namespace vx {

class IExecutable;
class Context;

class IExecutor {
  public:
   using task = std::weak_ptr<IExecutable>;
  virtual ~IExecutor(){
     std::cout<<"Destructor IExecutor\n";
  };
   virtual bool Submit(const std::shared_ptr<IExecutable> executable, const std::shared_ptr<IExecutable> ref, bool after=true) = 0;
   virtual bool Trigger(bool async=false) = 0;
   virtual std::shared_ptr<IExecutable> Compile(const std::shared_ptr<Graph>& graph) = 0;
   virtual std::shared_ptr<IDevice> Device() = 0;
   virtual std::shared_ptr<Context> Contex(){return context_;};

  protected:
   std::vector<task> tasks_;
   std::shared_ptr<IDevice> device_;
   std::shared_ptr<Context> context_;

};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_IEXECUTOR_H_*/
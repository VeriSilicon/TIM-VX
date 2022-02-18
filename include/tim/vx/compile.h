#ifndef TIM_VX_COMPILE_H_
#define TIM_VX_COMPILE_H_
#include "IExecutor.h"
#include "IExecutable.h"

namespace tim {
namespace vx {

   std::shared_ptr<IExecutable> Compile(std::shared_ptr<Graph> graph, std::shared_ptr<IExecutor> executor){
     return executor->Compile(graph);
   }

}  // namespace vx
}  // namespace tim

#endif

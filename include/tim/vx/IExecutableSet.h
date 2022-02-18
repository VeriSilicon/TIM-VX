#ifndef TIM_VX_IEXECUTABLE_SET_H_
#define TIM_VX_IEXECUTABLE_SET_H_
#include "IExecutable.h"

namespace tim {
namespace vx {
class IExecutable;
class IExecutableSet : public IExecutable{
 public:
  IExecutableSet(std::vector<std::shared_ptr<IExecutable>> executables);
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  bool Submit(std::shared_ptr<IExecutable> ref, bool after=true) override;
  bool Trigger(bool async=false) override;
  bool NBCompile() override;
  // std::shared_ptr<Graph> NBGraph() const {return executables_[0]->NBGraph();}
  std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec) override;
  std::vector<std::shared_ptr<IExecutable>> Executables();
  
 protected:
  std::vector<std::shared_ptr<IExecutable>> executables_;

};


}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_IEXECUTABLE_SET_H_*/
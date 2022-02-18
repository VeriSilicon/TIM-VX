#ifndef TIM_VX_LOCALIEXECUTABLE_H_
#define TIM_VX_LOCALIEXECUTABLE_H_
#include <memory>
#include "IExecutable.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/tensor.h"
#include "tim/vx/LocalTensorHandle.h"

namespace tim {
namespace vx {

class LocalExecutable : public IExecutable{
  public:
    LocalExecutable(std::shared_ptr<IExecutor>& executor, std::vector<char> nb_mem, size_t inputs, size_t outputs);
    ~LocalExecutable(){
      printf("Destructor LocalExecutable: %p\n", this);
    };
    void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
    void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
    bool Submit(std::shared_ptr<IExecutable> ref, bool after=true) override;
    bool Trigger(bool async=false) override;
    std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec);
    std::shared_ptr<Graph> NBGraph() const {return nb_graph_;}
    bool NBCompile() override;

  protected:
    std::shared_ptr<tim::vx::ops::NBG> nb_node_;
    std::vector<char> nb_buf_;

};

}  // namespace vx
}  // namespace tim
#endif /* TIM_VX_LOCALIEXECUTABLE_H_*/
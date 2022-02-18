#ifndef TIM_VX_ITENSORHANDLE_H_
#define TIM_VX_ITENSORHANDLE_H_
#include <memory>
#include "tensor.h"

namespace tim {
namespace vx {

class ITensorHandle {
 public:
  virtual ~ITensorHandle(){};
  virtual void CopyDataToTensor(const void* data, uint32_t size_in_bytes) = 0;
  virtual std::shared_ptr<Tensor> tensor() const = 0;

 protected:

};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_ITENSORHANDLE_H_*/
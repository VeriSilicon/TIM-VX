#ifndef TIM_VX_LOCALTENSORHANDLE_H_
#define TIM_VX_LOCALTENSORHANDLE_H_
#include <memory>
#include "ITensorHandle.h"

namespace tim {
namespace vx {

class LocalTensorHandle : public ITensorHandle {
 public:
  LocalTensorHandle(const std::shared_ptr<Tensor>& tensor);

  void CopyDataToTensor(const void* data, uint32_t size_in_bytes);

  std::shared_ptr<Tensor> tensor() const;

 protected:
  std::shared_ptr<Tensor> tensor_;

};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_LOCALTENSORHANDLE_H_*/
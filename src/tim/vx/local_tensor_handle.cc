# include "tim/vx/LocalTensorHandle.h"

namespace tim {
namespace vx {

LocalTensorHandle::LocalTensorHandle(const std::shared_ptr<Tensor>& tensor){
      tensor_ = tensor;
  }

  void LocalTensorHandle::CopyDataToTensor(const void* data, uint32_t size_in_bytes){
    tensor_->CopyDataToTensor(data, size_in_bytes);
  }

  std::shared_ptr<Tensor> LocalTensorHandle::tensor() const {
    return tensor_;
  }
}  // namespace vx
}  // namespace tim

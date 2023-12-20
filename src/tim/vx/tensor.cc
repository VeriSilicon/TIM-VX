/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include "tim/vx/tensor.h"

#include <algorithm>

#include "graph_private.h"
#include "tensor_private.h"
#include "tim/vx/graph.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

#ifndef ENABLE_TENSOR_HNDL
#define ENABLE_TENSOR_HNDL 1
#endif

namespace {

void PackTensorDtype(tim::vx::TensorSpec& spec, vsi_nn_dtype_t* dtype) {
  dtype->vx_type = TranslateDataType(spec.datatype_);
  dtype->qnt_type = TranslateQuantType(spec.quantization_.Type());
  switch (spec.quantization_.Type()) {
    case tim::vx::QuantType::NONE:
      break;
    case tim::vx::QuantType::ASYMMETRIC:
      dtype->scale = spec.quantization_.Scales()[0];
      dtype->zero_point = spec.quantization_.ZeroPoints()[0];
      //note:temporarily ignore the Uint8 weight case in conv.
      // if (dtype->vx_type == VSI_NN_TYPE_UINT8 && dtype->zero_point == 0) {
      //   dtype->vx_type = VSI_NN_TYPE_INT8;
      // }
      break;
    case tim::vx::QuantType::SYMMETRIC_PER_CHANNEL: {
      dtype->scales = spec.quantization_.Scales().data();
      dtype->scale_dim = spec.quantization_.ZeroPoints().size();
#if (VSI_NN_VERSION_MAJOR == 1 && VSI_NN_VERSION_MINOR == 1 && \
     VSI_NN_VERSION_PATCH <= 18)
      {
        std::vector<float> zps(spec.quantization_.ZeroPoints().size());
        std::transform(spec.quantization_.ZeroPoints().begin(),
                       spec.quantization_.ZeroPoints().end(), zps.begin(),
                       [](const int& it) { return static_cast<float>(it); });
        dtype->zero_points = zps.data();
      }
#else
      dtype->zero_points = spec.quantization_.ZeroPoints().data();
#endif
      dtype->zero_points_dim = spec.quantization_.ZeroPoints().size();
      dtype->channel_dim = spec.quantization_.ChannelDim();
      break;
    }

    case tim::vx::QuantType::DYNAMIC_FIXED_POINT:
      dtype->fl = spec.quantization_.Fl();
      break;
    default:
      break;
  }
}

}  // namespace
namespace tim {
namespace vx {

TensorImpl::TensorImpl(Graph* graph, const TensorSpec& spec, const void* data)
    : graph_(reinterpret_cast<GraphImpl*>(graph)),
      id_(VSI_NN_TENSOR_ID_NA),
      spec_(spec),
      data_(const_cast<void*>(data)) {
  Init();
  if (spec_.attr_ & (TensorAttribute::INPUT | TensorAttribute::OUTPUT)) {
    data_ = nullptr;  // it's not needed to reset it in a constant tensor
  }
}

TensorImpl::TensorImpl(Graph* graph, const TensorSpec& spec,
                       const DmaBufferDesc& dmafd)
    : graph_(reinterpret_cast<GraphImpl*>(graph)),
      id_(VSI_NN_TENSOR_ID_NA),
      spec_(spec),
      data_(nullptr),
      fd_(dmafd.fd) {
  Init();
}

TensorImpl::TensorImpl(Graph* graph, const TensorSpec& spec, void* data)
    : graph_(reinterpret_cast<GraphImpl*>(graph)),
      id_(VSI_NN_TENSOR_ID_NA),
      spec_(spec),
      data_(nullptr) {
  if (!(spec_.attr_ & (TensorAttribute::INPUT | TensorAttribute::OUTPUT))) {
    VSILOGE("TensorImpl with an external data got unexpected attr");
    return;
  }
  Init(data);
  data_ = data;
}

TensorImpl::~TensorImpl() {}

bool TensorImpl::SaveTensorToTextByFp32(std::string filename) {
  vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
  vsi_nn_SaveTensorToTextByFp32(graph_->graph(), tensor, filename.c_str(),
                                NULL);
  return true;
}

void* TensorImpl::ConvertTensorToData(uint8_t* tensorData) {
  vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
  tensorData = vsi_nn_ConvertTensorToData(graph_->graph(), tensor);
  return tensorData;
}

bool TensorImpl::CopyDataToTensor(const void* data, uint32_t size_in_bytes) {
  (void)size_in_bytes;
  if (!IsWriteable()) {
    return false;
  }

  bool retn = true;
  if (data && VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
    if (tensor) {
      uint32_t tensor_bytes = vsi_nn_GetTensorSize(
          tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type);

      if (tensor->attr.is_created_from_handle) {
        void* ptr = NULL;
        vsi_nn_GetTensorHandle(tensor, &ptr);
        if (ptr) {
          memcpy(ptr, data, tensor_bytes);
          vsi_nn_FlushHandle(tensor);
          retn = true;
        } else {
          VSILOGE("GetTensorHandle fail");
        }
      } else {
        /*
        argument `data` of vsi_nn_CopyDataToTensor is non-const
        convert it from const data to non-const, will be fixed in ovxlib
        */
        const uint8_t* end = static_cast<const uint8_t*>(data) + tensor_bytes;
        std::vector<uint8_t> data_copy(static_cast<const uint8_t*>(data), end);

        retn = (VSI_SUCCESS == vsi_nn_CopyDataToTensor(graph_->graph(), tensor,
                                                       data_copy.data()));
      }
    }
  }
  return retn;
}

bool TensorImpl::CopyDataFromTensor(void* data) {
  if (!IsReadable()) {
    return false;
  }

  bool retn = true;
  if (data && VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);

    if (tensor) {
      uint32_t tensor_bytes = vsi_nn_GetTensorSize(
          tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type);

      if (tensor->attr.is_created_from_handle) {
        void* ptr = NULL;
        vsi_nn_GetTensorHandle(tensor, &ptr);
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        vsi_nn_InvalidateHandle(tensor);
#endif
        if (ptr) {
          memcpy(data, ptr, tensor_bytes);
          retn = true;
        } else {
          VSILOGE("GetTensorHandle fail");
        }
      } else {
        vsi_nn_CopyTensorToBuffer(graph_->graph(), tensor,
                                  static_cast<uint8_t*>(data));
        retn = true;
      }
    }
  }
  return retn;
}

float* TensorImpl::ConvertTensorToFloat32Data() {
  return vsi_nn_ConvertTensorToFloat32Data(
      graph_->graph(), vsi_nn_GetTensor(graph_->graph(), id_));
}

void TensorImpl::SetScalar(int8_t is_scalar) {
  bool retn = vsi_nn_SetTensorIsScalar(vsi_nn_GetTensor(graph_->graph(), id_),is_scalar);
  if (retn != VSI_SUCCESS) {
    VSILOGE("Setting scalar fail!");
  }
  return;
}

bool TensorImpl::SwapHandle(void* new_ptr, bool is_new_ptr_malloc_by_ovxlib,
                            void** old_ptr) {
  bool retn = true;
  if (VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
    if (tensor && tensor->attr.is_created_from_handle) {
      retn = (VSI_SUCCESS == vsi_nn_SwapHandle(tensor, new_ptr,
                                               is_new_ptr_malloc_by_ovxlib,
                                               old_ptr));
      if (!retn) {
        VSILOGE("SwapHandle fail");
      }
    }
  }
  return retn;
}

bool TensorImpl::SwapHandle(std::shared_ptr<tim::vx::Tensor> tensor) {
  bool retn = true;
  if (VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor0 = vsi_nn_GetTensor(graph_->graph(), id_);
    vsi_nn_tensor_t* tensor1 =
        vsi_nn_GetTensor(graph_->graph(), tensor->GetId());
    if (tensor0 && tensor0->attr.is_created_from_handle && tensor1 &&
        tensor1->attr.is_created_from_handle) {
      retn = (VSI_SUCCESS == vsi_nn_SwapTensorHandle(tensor0, tensor1));
      if (!retn) {
        VSILOGE("SwapHandle fail");
      }
    }
  }
  return retn;
}

bool TensorImpl::SwapHandleWithCache(std::shared_ptr<tim::vx::Tensor> tensor) {
#ifdef VSI_SWAP_HANDLE_CACHE_SUPPORT
  bool retn = true;
  if (VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor0 = vsi_nn_GetTensor(graph_->graph(), id_);
    vsi_nn_tensor_t* tensor1 =
        vsi_nn_GetTensor(graph_->graph(), tensor->GetId());
    if (tensor0 && tensor0->attr.is_created_from_handle && tensor1 &&
        tensor1->attr.is_created_from_handle) {
      retn = (VSI_SUCCESS == vsi_nn_SwapTensorHandleWithCache(graph_->graph(), tensor0, tensor1));
      if (!retn) {
        VSILOGE("SwapHandle fail");
      }
    }
  }
  return retn;
#else
  (void)tensor;
  VSILOGE("Your ovxlib DO NOT support vsi_nn_SwapTensorHandleWithCache API");
  return false;
#endif
}

bool TensorImpl::FlushCacheForHandle() {
  if (!(spec_.attr_ & TensorAttribute::INPUT)) {
    return false;
  }

  bool retn = true;
  if (VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
    if (tensor && tensor->attr.is_created_from_handle) {
      retn = (VSI_SUCCESS == vsi_nn_FlushHandle(tensor));
      if (!retn) {
        VSILOGE("FlushHandle fail");
      }
    }
  }
  return retn;
}

bool TensorImpl::InvalidateCacheForHandle() {
  if (!(spec_.attr_ & TensorAttribute::OUTPUT)) {
    return false;
  }

  bool retn = true;
  if (VSI_NN_TENSOR_ID_NA != id_) {
    retn = false;
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
    if (tensor && tensor->attr.is_created_from_handle) {
      void* ptr = NULL;
      retn = (VSI_SUCCESS == vsi_nn_GetTensorHandle(tensor, &ptr));
      if (!retn) {
        VSILOGE("GetTensorHandle fail");
      }
    }
  }
  return retn;
}

void* TensorImpl::map(bool invalidate_cpu_cache) {
  if (!(spec_.attr_ & (TensorAttribute::INPUT | TensorAttribute::OUTPUT))) {
    return nullptr;
  }

  void* cpu_ptr = nullptr;
  if (VSI_NN_TENSOR_ID_NA != id_) {
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
    if (tensor && tensor->attr.is_created_from_handle) {
      // Here `cpu_cache` means L1/L2/... cache on a CPU chip.
      // If data_ has been updated by other devices like NPU,
      // then caches on CPU MUST be invalidated before reading.
      if (data_ && !invalidate_cpu_cache) {
        cpu_ptr = data_;
      } else {
        vsi_nn_GetTensorHandle(tensor, &cpu_ptr);
        // TODO: what to do when fd_ != -1
      }
      if (!cpu_ptr) {
        VSILOGE("GetTensorHandle fail");
      }
    }
  }
  return cpu_ptr;
}

void TensorImpl::unmap() {
  if (!(spec_.attr_ & (TensorAttribute::INPUT | TensorAttribute::OUTPUT))) {
    return;
  }
  if (VSI_NN_TENSOR_ID_NA == id_) {
    return;
  }
  if (-1 == fd_) {
    if (data_ && spec_.attr_ & TensorAttribute::INPUT) {
      // Here data_ is an external buffer and may have been updated
      vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_->graph(), id_);
      if (tensor && tensor->attr.is_created_from_handle) {
        bool retn = (VSI_SUCCESS == vsi_nn_FlushHandle(tensor));
        if (!retn) {
          VSILOGE("FlushHandle fail");
        }
      }
    }
    return;
  }
  // TODO: unmap fd_
}

bool TensorImpl::Init(void* external_cache) {
  vsi_nn_tensor_attr_t attr;

#if (!ENABLE_TENSOR_HNDL)
  (void)external_cache;
#endif

  memset(&attr, 0x00, sizeof(attr));
  attr.dim_num = spec_.shape_.size();
  attr.is_const = static_cast<bool>(spec_.attr_ & TensorAttribute::CONSTANT);
  attr.vtl = static_cast<bool>(spec_.attr_ & TensorAttribute::TRANSIENT);

  // Use auto shape for virtual tensors so that tim-vx can perform it's own
  // shape inference
  if (attr.vtl) {
    attr.dim_num = VSI_NN_DIM_AUTO;
  }

  for (ShapeType::size_type i = 0; i < spec_.shape_.size(); i++) {
    attr.size[i] = spec_.shape_[i];
  }

  PackTensorDtype(spec_, &attr.dtype);

#if (ENABLE_TENSOR_HNDL)
  if ((spec_.attr_ & TensorAttribute::INPUT) ||
      (spec_.attr_ & TensorAttribute::OUTPUT)) {
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
    if (fd_ != -1) {
      attr.vsi_memory_type = VSI_MEMORY_TYPE_DMABUF;
    }

    id_ = vsi_nn_AddTensorFromHandle(
        graph_->graph(),
        VSI_NN_TENSOR_ID_AUTO,  // DMABUF's fd is created by TensorFromHandle as input or output,
        &attr,
        fd_ != -1 ? (uint8_t*)fd_
                  : (uint8_t*)external_cache);  // and cannot be set to const
#else
    if (-1 == fd_) {
      id_ = vsi_nn_AddTensorFromHandle(graph_->graph(), VSI_NN_TENSOR_ID_AUTO,
                                       &attr, (uint8_t*)external_cache);
    } else {
      id_ = 0xFFFFFFFF;
      VSILOGE("Create tensor fail: low-level driver doesn't support dmabuffer");
    }
#endif

  } else
#endif
  {
    id_ = vsi_nn_AddTensor(graph_->graph(), VSI_NN_TENSOR_ID_AUTO, &attr,
                           nullptr);
  }

  if (VSI_NN_TENSOR_ID_NA == id_) {
    VSILOGE("Create tensor fail!");
    return false;
  }

  if (data_) {
    if (!CopyDataToTensor(data_, 0)) {
      VSILOGE("Copy data to tensor fail!");
      return false;
    }
  }

  return true;
}

uint32_t TensorImpl::GetId() { return id_; }

bool TensorImpl::IsWriteable() {
  return spec_.attr_ != TensorAttribute::TRANSIENT;
}

bool TensorImpl::IsReadable() {
  return spec_.attr_ != TensorAttribute::TRANSIENT;
}

TensorSpec::TensorSpec(const TensorSpec& other) {
  this->datatype_ = other.datatype_;
  this->shape_ = other.shape_;
  this->attr_ = other.attr_;
  this->quantization_ = other.quantization_;
}

TensorSpec& TensorSpec::operator=(const TensorSpec& other) {
  this->datatype_ = other.datatype_;
  this->shape_ = other.shape_;
  this->attr_ = other.attr_;
  this->quantization_ = other.quantization_;
  return *this;
}

TensorSpec& TensorSpec::SetDataType(DataType datatype) {
  this->datatype_ = datatype;
  return *this;
}

TensorSpec& TensorSpec::SetShape(const ShapeType& shape) {
  this->shape_ = shape;
  return *this;
}

TensorSpec& TensorSpec::SetAttribute(TensorAttribute attr) {
  this->attr_ = attr;
  return *this;
}

TensorSpec& TensorSpec::SetQuantization(Quantization& quantization) {
  this->quantization_ = quantization;
  return *this;
}

TensorSpec TensorSpec::AsTransientSpec() const {
  return TensorSpec(this->datatype_, ShapeType({}), TensorAttribute::TRANSIENT,
                    this->quantization_);
}

int64_t TensorSpec::GetElementNum() const {
  int64_t count = 1;
  for (auto dim : shape_) {
    count *= dim;
  }
  return count;
}

int64_t TensorSpec::GetElementByteSize() const {
  switch (datatype_) {
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::BOOL8:
      return 1;
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FLOAT16:
      return 2;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      return 4;
    default:
      return 1;
  }
}

int64_t TensorSpec::GetByteSize() const {
  return GetElementNum() * GetElementByteSize();
}

bool TensorSpec::operator==(const TensorSpec& other_spec) const {
  if (datatype_ == other_spec.datatype_ && shape_ == other_spec.shape_ &&
      attr_ == other_spec.attr_ && quantization_ == other_spec.quantization_) {
    return true;
  } else {
    return false;
  }
}

bool Quantization::operator==(const Quantization& other_quant) const {
  if (type_ != tim::vx::QuantType::DYNAMIC_FIXED_POINT) {
    if (type_ == other_quant.type_ && scales_ == other_quant.scales_ &&
        zero_points_ == other_quant.zero_points_ &&
        channel_dim_ == other_quant.channel_dim_)
      return true;
  } else if (fl_ == other_quant.fl_)
    return true;
  return false;
}

namespace utils {
bool Float32ToDtype(std::shared_ptr<tim::vx::Tensor> tensor,
                    std::vector<float> fval, uint8_t* tensorData) {
  bool retn = true;
  vsi_nn_tensor_attr_t attr;
  uint32_t sz = tensor->GetSpec().GetElementNum();
  uint32_t stride = tensor->GetSpec().GetElementByteSize();
  PackTensorDtype(tensor->GetSpec(), &attr.dtype);
  for (uint32_t i = 0; i < sz; i++) {
    retn = (VSI_SUCCESS == vsi_nn_Float32ToDtype(
                               fval[i], &tensorData[i * stride], &attr.dtype));
    if (!retn) {
      VSILOGE("Convert data fail");
      return retn;
    }
  }
  return retn;
}

bool DtypeToFloat32(std::shared_ptr<tim::vx::Tensor> tensor,
                    uint8_t* tensorData, float* data) {
  bool retn = true;
  vsi_nn_tensor_attr_t attr;

  PackTensorDtype(tensor->GetSpec(), &attr.dtype);
  retn = (VSI_SUCCESS == vsi_nn_DtypeToFloat32(tensorData, data, &attr.dtype));
  return retn;
}
}  //namespace utils
}  // namespace vx
}  // namespace tim

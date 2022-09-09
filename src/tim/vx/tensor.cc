/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
      data_(const_cast<void *>(data)) {
  Init();
  if (spec_.attr_ & (TensorAttribute::INPUT | TensorAttribute::OUTPUT)) {
    data_ = nullptr; // it's not needed to reset it in a constant tensor
  }
}

TensorImpl::TensorImpl(Graph* graph, const TensorSpec& spec, const DmaBufferDesc& dmafd)
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
        void *ptr = NULL;
        vsi_nn_GetTensorHandle(tensor, &ptr);
        if (ptr) {
          memcpy(ptr, data, tensor_bytes);
          vsi_nn_FlushHandle(tensor);
          retn = true;
        } else {
          VSILOGE("GetTensorHandle fail");
        }
      }
      else {
        /*
        argument `data` of vsi_nn_CopyDataToTensor is non-const
        convert it from const data to non-const, will be fixed in ovxlib
        */
        const uint8_t* end = static_cast<const uint8_t*>(data) + tensor_bytes;
        std::vector<uint8_t> data_copy(static_cast<const uint8_t*>(data), end);

        retn = (VSI_SUCCESS ==
             vsi_nn_CopyDataToTensor(graph_->graph(), tensor, data_copy.data()));
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
      if ( tensor && tensor->attr.is_created_from_handle) {
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

bool TensorImpl::Init(void *external_cache) {
  vsi_nn_tensor_attr_t attr;

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
        fd_ != -1 ? (uint8_t*)fd_ : (uint8_t*)external_cache); // and cannot be set to const
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

TensorSpec& TensorSpec::SetShape(ShapeType& shape) {
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

}  // namespace vx
}  // namespace tim

/****************************************************************************
*
*    Copyright (c) 2023 Vivante Corporation
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
#include "grpc_platform_client.h"

namespace {
::rpc::DataType MapDataType(tim::vx::DataType type) {
  ::rpc::DataType rpc_type;
  switch (type) {
    case tim::vx::DataType::FLOAT32:
      rpc_type = ::rpc::DataType::FLOAT32;
      break;
    case tim::vx::DataType::FLOAT16:
      rpc_type = ::rpc::DataType::FLOAT16;
      break;
    case tim::vx::DataType::INT64:
      rpc_type = ::rpc::DataType::INT64;
      break;
    case tim::vx::DataType::INT32:
      rpc_type = ::rpc::DataType::INT32;
      break;
    case tim::vx::DataType::INT16:
      rpc_type = ::rpc::DataType::INT16;
      break;
    case tim::vx::DataType::INT8:
      rpc_type = ::rpc::DataType::INT8;
      break;
    case tim::vx::DataType::UINT32:
      rpc_type = ::rpc::DataType::UINT32;
      break;
    case tim::vx::DataType::UINT16:
      rpc_type = ::rpc::DataType::UINT16;
      break;
    case tim::vx::DataType::UINT8:
      rpc_type = ::rpc::DataType::UINT8;
      break;
    case tim::vx::DataType::BOOL8:
      rpc_type = ::rpc::DataType::BOOL8;
      break;
    default:
      std::cout << "unknown tim vx data type" << std::endl;
      assert(false);
  }
  return rpc_type;
}

::rpc::TensorAttr MapTensorAttr(tim::vx::TensorAttribute attr) {
  ::rpc::TensorAttr rpc_attr;
  switch (attr) {
    case tim::vx::TensorAttribute::INPUT:
      rpc_attr = ::rpc::TensorAttr::INPUT;
      break;
    case tim::vx::TensorAttribute::OUTPUT:
      rpc_attr = ::rpc::TensorAttr::OUTPUT;
      break;
    default:
      std::cout << "invalid tim vx tensor attr" << std::endl;
      assert(false);
  }
  return rpc_attr;
}

::rpc::QuantType MapQuantType(tim::vx::QuantType quant) {
  ::rpc::QuantType rpc_quant;
  switch (quant) {
    case tim::vx::QuantType::NONE:
      rpc_quant = ::rpc::QuantType::NONE;
      break;
    case tim::vx::QuantType::ASYMMETRIC:
      rpc_quant = ::rpc::QuantType::ASYMMETRIC;
      break;
    case tim::vx::QuantType::SYMMETRIC_PER_CHANNEL:
      rpc_quant = ::rpc::QuantType::SYMMETRIC_PER_CHANNEL;
      break;
    default:
      std::cout << "invalid tim vx quant type" << std::endl;
      assert(false);
  }
  return rpc_quant;
}

}  // namespace
namespace tim {
namespace vx {
namespace platform {
int32_t GRPCPlatformClient::Enumerate() {
  ::grpc::ClientContext context;
  ::rpc::EmptyMsg emsg;
  ::rpc::DeviceCount device_count;
  stub_->Enumerate(&context, emsg, &device_count);

  return device_count.count();
}

int32_t GRPCPlatformClient::CreateExecutor(int32_t device) {
  ::grpc::ClientContext context;
  ::rpc::Device device_msg;
  device_msg.set_device(device);
  ::rpc::Executor executor_msg;
  stub_->CreateExecutor(&context, device_msg, &executor_msg);

  return executor_msg.executor();
}

int32_t GRPCPlatformClient::CreateExecutable(int32_t executor,
                                             const std::vector<char>& nbg,
                                             int32_t input_size,
                                             int32_t output_size) {
  ::grpc::ClientContext context;
  ::rpc::GraphInfo graph_info_msg;
  graph_info_msg.set_executor(executor);
  graph_info_msg.set_input_size(input_size);
  graph_info_msg.set_output_size(output_size);
  std::string nbg_str(nbg.data(), nbg.size());
  graph_info_msg.set_nbg(nbg_str);
  ::rpc::Executable executable_msg;
  stub_->CreateExecutable(&context, graph_info_msg, &executable_msg);

  return executable_msg.executable();
}

int32_t GRPCPlatformClient::AllocateTensor(int32_t executable,
                                           const tim::vx::TensorSpec& spec) {
  ::grpc::ClientContext context;
  ::rpc::TensorInfo tensor_info_msg;
  ::rpc::Tensor tensor_msg;
  tensor_info_msg.set_executable(executable);
  tensor_info_msg.mutable_tensor_spec()->set_data_type(
      MapDataType(spec.datatype_));
  tensor_info_msg.mutable_tensor_spec()->set_tensor_attr(
      MapTensorAttr(spec.attr_));
  tensor_info_msg.mutable_tensor_spec()->mutable_quant()->set_quant_type(
      MapQuantType(spec.quantization_.Type()));
  for (uint32_t s : spec.shape_) {
    tensor_info_msg.mutable_tensor_spec()->add_shape(s);
  }

  stub_->AllocateTensor(&context, tensor_info_msg, &tensor_msg);
  return tensor_msg.tensor();
}

bool GRPCPlatformClient::SetInput(int32_t executable, int32_t tensor) {
  ::grpc::ClientContext context;
  ::rpc::IOTensor in_tensor_msg;
  ::rpc::Status status_msg;
  in_tensor_msg.set_executable(executable);
  in_tensor_msg.set_tensor(tensor);

  stub_->SetInput(&context, in_tensor_msg, &status_msg);
  return status_msg.status();
}

bool GRPCPlatformClient::SetOutput(int32_t executable, int32_t tensor) {
  ::grpc::ClientContext context;
  ::rpc::IOTensor out_tensor_msg;
  ::rpc::Status status_msg;
  out_tensor_msg.set_executable(executable);
  out_tensor_msg.set_tensor(tensor);

  stub_->SetOutput(&context, out_tensor_msg, &status_msg);
  return status_msg.status();
}

bool GRPCPlatformClient::Submit(int32_t executable) {
  ::grpc::ClientContext context;
  ::rpc::Executable executable_mag;
  ::rpc::Status status_msg;
  executable_mag.set_executable(executable);

  stub_->Submit(&context, executable_mag, &status_msg);
  return status_msg.status();
}

bool GRPCPlatformClient::Trigger(int32_t executor) {
  ::grpc::ClientContext context;
  ::rpc::Executor executor_mag;
  ::rpc::Status status_msg;
  executor_mag.set_executor(executor);

  stub_->Trigger(&context, executor_mag, &status_msg);
  return status_msg.status();
}

bool GRPCPlatformClient::CopyDataToTensor(int32_t tensor, const void* data,
                                          int32_t length) {
  ::grpc::ClientContext context;
  ::rpc::TensorData tensor_data_msg;
  ::rpc::Status status_msg;
  tensor_data_msg.set_tensor(tensor);
  std::string data_str(reinterpret_cast<const char*>(data), length);
  tensor_data_msg.set_data(data_str);

  stub_->CopyDataToTensor(&context, tensor_data_msg, &status_msg);
  return status_msg.status();
}

bool GRPCPlatformClient::CopyDataFromTensor(int32_t tensor, void* data) {
  ::grpc::ClientContext context;
  ::rpc::Tensor tensor_msg;
  ::rpc::Data data_msg;
  ::rpc::Status status_msg;
  tensor_msg.set_tensor(tensor);

  stub_->CopyDataFromTensor(&context, tensor_msg, &data_msg);
  std::string data_str = data_msg.data();
  memcpy(data, data_str.data(), data_str.size());
  return (data != nullptr);
}

void GRPCPlatformClient::Clean() {
  ::grpc::ClientContext context;
  ::rpc::EmptyMsg emsg;
  ::rpc::Status status_msg;

  stub_->Clean(&context, emsg, &status_msg);
}

}  // namespace platform
}  // namespace vx
}  // namespace tim

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
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "grpc_platform.grpc.pb.h"
#include "tim/vx/platform/native.h"
#include "vsi_nn_pub.h"
#ifdef ENABLE_PLATFORM_LITE
#include "tim/vx/platform/lite/lite_native.h"
#endif

std::unordered_map<int32_t, std::shared_ptr<tim::vx::platform::IDevice>>
    device_table;
std::unordered_map<int32_t, std::shared_ptr<tim::vx::platform::IExecutor>>
    executor_table;
std::vector<std::shared_ptr<tim::vx::platform::IExecutable>> executable_table;
std::vector<std::shared_ptr<tim::vx::platform::ITensorHandle>> tensor_table;

namespace {
tim::vx::DataType MapDataType(::rpc::DataType type) {
  tim::vx::DataType vx_type;
  switch (type) {
    case ::rpc::DataType::FLOAT32:
      vx_type = tim::vx::DataType::FLOAT32;
      break;
    case ::rpc::DataType::FLOAT16:
      vx_type = tim::vx::DataType::FLOAT16;
      break;
    case ::rpc::DataType::INT64:
      vx_type = tim::vx::DataType::INT64;
      break;
    case ::rpc::DataType::INT32:
      vx_type = tim::vx::DataType::INT32;
      break;
    case ::rpc::DataType::INT16:
      vx_type = tim::vx::DataType::INT16;
      break;
    case ::rpc::DataType::INT8:
      vx_type = tim::vx::DataType::INT8;
      break;
    case ::rpc::DataType::UINT32:
      vx_type = tim::vx::DataType::UINT32;
      break;
    case ::rpc::DataType::UINT16:
      vx_type = tim::vx::DataType::UINT16;
      break;
    case ::rpc::DataType::UINT8:
      vx_type = tim::vx::DataType::UINT8;
      break;
    case ::rpc::DataType::BOOL8:
      vx_type = tim::vx::DataType::BOOL8;
      break;
    default:
      std::cout << "unknown data type" << std::endl;
      assert(false);
  }
  return vx_type;
}

tim::vx::TensorAttribute MapTensorAttr(::rpc::TensorAttr attr) {
  tim::vx::TensorAttribute vx_attr;
  switch (attr) {
    case ::rpc::TensorAttr::INPUT:
      vx_attr = tim::vx::TensorAttribute::INPUT;
      break;
    case ::rpc::TensorAttr::OUTPUT:
      vx_attr = tim::vx::TensorAttribute::OUTPUT;
      break;
    default:
      std::cout << "invalid tensor attr" << std::endl;
      assert(false);
  }
  return vx_attr;
}

tim::vx::QuantType MapQuantType(::rpc::QuantType quant) {
  tim::vx::QuantType vx_quant;
  switch (quant) {
    case ::rpc::QuantType::NONE:
      vx_quant = tim::vx::QuantType::NONE;
      break;
    case ::rpc::QuantType::ASYMMETRIC:
      vx_quant = tim::vx::QuantType::ASYMMETRIC;
      break;
    case ::rpc::QuantType::SYMMETRIC_PER_CHANNEL:
      vx_quant = tim::vx::QuantType::SYMMETRIC_PER_CHANNEL;
      break;
    default:
      std::cout << "invalid quant type" << std::endl;
      assert(false);
  }
  return vx_quant;
}
}  // namespace

class GRPCPlatformService final : public ::rpc::GRPCPlatform::Service {
 public:
  ::grpc::Status Enumerate(::grpc::ServerContext* context,
                           const ::rpc::EmptyMsg* request,
                           ::rpc::DeviceCount* response) override {
    VSILOGD("------ Calling gRPC Enumerate ------");
    (void)context;
    (void)request;
    auto devices = tim::vx::platform::NativeDevice::Enumerate();
    response->set_count(devices.size());
    for (int i = 0; i < static_cast<int>(devices.size()); ++i) {
      device_table.insert({i, devices[i]});
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status CreateExecutor(::grpc::ServerContext* context,
                                const ::rpc::Device* request,
                                ::rpc::Executor* response) override {
    VSILOGD("------ Calling gRPC CreateExecutor ------");
    (void)context;
    int32_t id = request->device();
    auto device = device_table[id];
#ifdef ENABLE_PLATFORM_LITE
    auto executor =
        std::make_shared<tim::vx::platform::LiteNativeExecutor>(device);
#else
    auto executor = std::make_shared<tim::vx::platform::NativeExecutor>(device);
#endif
    executor_table.insert({id, executor});
    response->set_executor(id);
    return ::grpc::Status::OK;
  }

  ::grpc::Status CreateExecutable(::grpc::ServerContext* context,
                                  const ::rpc::GraphInfo* request,
                                  ::rpc::Executable* response) override {
    VSILOGD("------ Calling gRPC CreateExecutable ------");
    (void)context;
    int32_t id = request->executor();
    auto executor = executor_table[id];
    std::string nbg_str = request->nbg();
    std::vector<char> nbg_vec(nbg_str.size());
    memcpy(nbg_vec.data(), nbg_str.data(), nbg_str.size());
#ifdef ENABLE_PLATFORM_LITE
    auto executable = std::make_shared<tim::vx::platform::LiteNativeExecutable>(
        executor, nbg_vec);
#else
    int32_t input_size = request->input_size();
    int32_t output_size = request->output_size();
    auto executable = std::make_shared<tim::vx::platform::NativeExecutable>(
        executor, nbg_vec, input_size, output_size);
#endif
    executable_table.push_back(executable);
    response->set_executable(executable_table.size() - 1);
    return ::grpc::Status::OK;
  }

  ::grpc::Status AllocateTensor(::grpc::ServerContext* context,
                                const ::rpc::TensorInfo* request,
                                ::rpc::Tensor* response) override {
    VSILOGD("------ Calling gRPC AllocateTensor ------");
    (void)context;
    int32_t id = request->executable();
    auto executable = executable_table[id];
    tim::vx::DataType data_type =
        MapDataType(request->tensor_spec().data_type());
    tim::vx::TensorAttribute tensor_attr =
        MapTensorAttr(request->tensor_spec().tensor_attr());
    tim::vx::QuantType quant_type =
        MapQuantType(request->tensor_spec().quant().quant_type());
    auto shape = request->tensor_spec().shape();
    tim::vx::ShapeType vx_shape(shape.size());
    for (int i = 0; i < shape.size(); ++i) vx_shape[i] = shape[i];
    tim::vx::TensorSpec tensor_spec;
    if (quant_type == tim::vx::QuantType::NONE) {
      tensor_spec = tim::vx::TensorSpec(data_type, vx_shape, tensor_attr);
    } else {
      tim::vx::Quantization quantization;
      quantization.SetType(quant_type);
      quantization.SetChannelDim(request->tensor_spec().quant().channel_dim());
      auto scales = request->tensor_spec().quant().scales();
      auto zero_pionts = request->tensor_spec().quant().zero_points();
      std::vector<float> vx_scales(scales.size());
      std::vector<int32_t> vx_zero_points(zero_pionts.size());
      for (int i = 0; i < scales.size(); ++i) vx_scales[i] = scales[i];
      for (int i = 0; i < zero_pionts.size(); ++i) {
        vx_zero_points[i] = zero_pionts[i];
      }
      quantization.SetScales(vx_scales);
      quantization.SetZeroPoints(vx_zero_points);

      tensor_spec =
          tim::vx::TensorSpec(data_type, vx_shape, tensor_attr, quantization);
    }

    auto tensor_handle = executable->AllocateTensor(tensor_spec);
    tensor_table.push_back(tensor_handle);
    response->set_tensor(tensor_table.size() - 1);

    return ::grpc::Status::OK;
  }

  ::grpc::Status SetInput(::grpc::ServerContext* context,
                          const ::rpc::IOTensor* request,
                          ::rpc::Status* response) override {
    VSILOGD("------ Calling gRPC SetInput ------");
    (void)context;
    int32_t tensor_id = request->tensor();
    int32_t executable_id = request->executable();
    auto executable = executable_table[executable_id];
    auto tensor_handle = tensor_table[tensor_id];
    if (tensor_handle->GetTensor()->GetSpec().attr_ !=
        tim::vx::TensorAttribute::INPUT) {
      VSILOGE("You are setting a no-input tensor as graph input");
    }
    executable->SetInput(tensor_handle);
    response->set_status(true);
    return ::grpc::Status::OK;
  }

  ::grpc::Status SetOutput(::grpc::ServerContext* context,
                           const ::rpc::IOTensor* request,
                           ::rpc::Status* response) override {
    VSILOGD("------ Calling gRPC SetOutput ------");
    (void)context;
    int32_t tensor_id = request->tensor();
    int32_t executable_id = request->executable();
    auto executable = executable_table[executable_id];
    auto tensor_handle = tensor_table[tensor_id];
    if (tensor_handle->GetTensor()->GetSpec().attr_ !=
        tim::vx::TensorAttribute::OUTPUT) {
      VSILOGE("You are setting a no-output tensor as graph output");
    }
    executable->SetOutput(tensor_handle);
    response->set_status(true);
    return ::grpc::Status::OK;
  }

  ::grpc::Status Submit(::grpc::ServerContext* context,
                        const ::rpc::Executable* request,
                        ::rpc::Status* response) override {
    VSILOGD("------ Calling gRPC Submit ------");
    (void)context;
    int32_t id = request->executable();
    auto executable = executable_table[id];
    executable->Submit(executable);
    response->set_status(true);
    return ::grpc::Status::OK;
  }

  ::grpc::Status Trigger(::grpc::ServerContext* context,
                         const ::rpc::Executor* request,
                         ::rpc::Status* response) override {
    VSILOGD("------ Calling gRPC Trigger ------");
    (void)context;
    int32_t id = request->executor();
    auto executor = executor_table[id];
    executor->Trigger();
    response->set_status(true);
    return ::grpc::Status::OK;
  }

  ::grpc::Status CopyDataToTensor(::grpc::ServerContext* context,
                                  const ::rpc::TensorData* request,
                                  ::rpc::Status* response) override {
    VSILOGD("------ Calling gRPC CopyDataToTensor ------");
    (void)context;
    int32_t id = request->tensor();
    auto tensor_handle = tensor_table[id];
    std::string data_str = request->data();
    bool status =
        tensor_handle->CopyDataToTensor(data_str.data(), data_str.size());
    response->set_status(status);
    return ::grpc::Status::OK;
  }

  ::grpc::Status CopyDataFromTensor(::grpc::ServerContext* context,
                                    const ::rpc::Tensor* request,
                                    ::rpc::Data* response) override {
    VSILOGD("------ Calling gRPC CopyDataFromTensor ------");
    (void)context;
    int32_t id = request->tensor();
    auto tensor_handle = tensor_table[id];
    size_t data_size = tensor_handle->GetTensor()->GetSpec().GetByteSize();
    void* ptr = malloc(data_size);
    bool status = tensor_handle->CopyDataFromTensor(ptr);
    if (!status) {
      VSILOGE("CopyDataFromTensor fail");
      free(ptr);
      return ::grpc::Status::CANCELLED;
    }
    std::string data_str(reinterpret_cast<char*>(ptr), data_size);
    response->set_data(std::move(data_str));
    free(ptr);
    return ::grpc::Status::OK;
  }

  ::grpc::Status Clean(::grpc::ServerContext* context,
                       const ::rpc::EmptyMsg* request,
                       ::rpc::Status* response) override {
    VSILOGD("------ Calling gRPC Clean ------");
    (void)context;
    (void)request;
    executor_table.clear();
    executable_table.clear();
    tensor_table.clear();
    response->set_status(true);
    return ::grpc::Status::OK;
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "error: need a port to connect." << std::endl;
    return -1;
  }
  std::string port(argv[1]);
  GRPCPlatformService service;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(port, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << port << std::endl;
  server->Wait();

  return 0;
}
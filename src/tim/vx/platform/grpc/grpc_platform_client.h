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
#ifndef _GRPC_PLATFORM_CLIENT_
#define _GRPC_PLATFORM_CLIENT_

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/vx/types.h"
#include "grpc_platform.grpc.pb.h"

namespace tim {
namespace vx {
namespace platform {
class GRPCPlatformClient {
 public:
  GRPCPlatformClient(const std::string& port)
      : stub_(rpc::GRPCPlatform::NewStub(
            grpc::CreateChannel(port, grpc::InsecureChannelCredentials()))) {}

  int32_t Enumerate();

  int32_t CreateExecutor(int32_t device);

  int32_t CreateExecutable(int32_t executor, const std::vector<char>& nbg,
                           int32_t input_size, int32_t output_size);

  int32_t AllocateTensor(int32_t executable, const tim::vx::TensorSpec& spec);

  bool SetInput(int32_t executable, int32_t tensor);

  bool SetOutput(int32_t executable, int32_t tensor);

  bool Submit(int32_t executable);

  bool Trigger(int32_t executor);

  bool CopyDataToTensor(int32_t tensor, const void* data, int32_t length);

  bool CopyDataFromTensor(int32_t tensor, void* data);

  void Clean();

 private:
  std::unique_ptr<rpc::GRPCPlatform::Stub> stub_;
};
}  // namespace platform
}  // namespace vx
}  // namespace tim
#endif
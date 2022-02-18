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
#include "local_device_private.h"
#include "graph_private.h"

namespace tim {
namespace vx {

LocalDeviceImpl::LocalDeviceImpl(device_id_t id){
  vip_device_ = std::make_shared<vip::IDevice> (id);
  device_id_ = id;
}

bool LocalDeviceImpl::Submit(const std::shared_ptr<Graph> graph/*, ovxlib::func_t func=NULL, ovxlib::data_t data=NULL*/) {
  GraphImpl* graphimp= dynamic_cast<GraphImpl*> (graph.get()); // hack to downcast
  vsi_graph_v_.push_back(graphimp->graph());
  return true;
}

  bool LocalDeviceImpl::Trigger(bool async, async_callback cb) {
    // extract graph from graph_tasks
    (void)async;
    bool status = false;
    while(!vsi_graph_v_.empty()){
      auto task = vsi_graph_v_.front();
      vsi_graph_v_.erase(vsi_graph_v_.begin());
      status = vip_device_->GraphSubmit(task, cb, NULL);
    }
    return status;
  }

  void LocalDeviceImpl::WaitDeviceIdle() {
    vip_device_->WaitThreadIdle();
  }

  bool LocalDeviceImpl::DeviceExit() {
    return vip_device_->ThreadExit();
  }

  std::vector<std::shared_ptr<IDevice>> LocalDevice::Enumerate(){
    std::vector<std::shared_ptr<IDevice>> device_v;
    device_id_t deviceCount = 0;
    vsi_nn_context_t context;
    context = vsi_nn_CreateContext();
    vxQueryContext(context->c, VX_CONTEXT_DEVICE_COUNT_VIV, &deviceCount, sizeof(deviceCount));
    std::cout<< "Device count = "<< deviceCount <<std::endl;
    for (device_id_t i = 0; i < deviceCount; i++){
      IDevice* local_device = new LocalDeviceImpl(i);
      std::shared_ptr<IDevice> local_device_sp(local_device);
      device_v.push_back(local_device_sp);
    }
    vsi_nn_ReleaseContext(&context);
    return device_v;
  }

}  // namespace vx
}  // namespace tim

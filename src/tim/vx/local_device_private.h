#ifndef TIM_VX_LOCAL_DEVICE_PRIVATE_H_
#define TIM_VX_LOCAL_DEVICE_PRIVATE_H_
#include <iostream>
#include "tim/vx/LocalDevice.h"
#include "vip/virtual_device.h"

namespace tim {
namespace vx {

class LocalDeviceImpl : public LocalDevice {
 public:
  LocalDeviceImpl(device_id_t id);
  ~LocalDeviceImpl(){
      printf("Destructor LocalDeviceImpl: %p\n", this);
    };

  bool Submit(const std::shared_ptr<Graph> graph/*, func_t func=NULL, data_t data=NULL*/) override;
  bool Trigger(bool async = false, async_callback cb = NULL) override;
  bool DeviceExit() override;
  void WaitDeviceIdle() override;

 protected:
 std::shared_ptr<vip::IDevice> vip_device_;
 std::vector<vsi_nn_graph_t*> vsi_graph_v_;

};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_LOCAL_DEVICE_PRIVATE_H_*/
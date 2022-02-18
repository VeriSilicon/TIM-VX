#ifndef TIM_VX_LOCAL_DEVICE_H_
#define TIM_VX_LOCAL_DEVICE_H_
#include "IDevice.h"

namespace tim {
namespace vx {

class LocalDevice : public IDevice {
 public:
  ~LocalDevice(){
      printf("Destructor LocalDevice: %p\n", this);
    };
  virtual bool Submit(const std::shared_ptr<Graph> graph/*, func_t func=NULL, data_t data=NULL*/) = 0;
  virtual bool Trigger(bool async = false, async_callback cb = NULL) = 0;
  virtual bool DeviceExit() = 0;
  virtual void WaitDeviceIdle() = 0;
  static std::vector<std::shared_ptr<IDevice>> Enumerate();

 protected:

};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_LOCAL_DEVICE_H_*/
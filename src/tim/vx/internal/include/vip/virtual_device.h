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
#ifndef _VIP_VIRTUAL_DEVICE_H
#define _VIP_VIRTUAL_DEVICE_H

#include <memory>
#include <functional>

struct _vsi_nn_graph;
typedef struct _vsi_nn_graph vsi_nn_graph_t;

namespace vip {

class Device;
using func_t = std::function<bool (const void*)>;
using data_t = const void*;

class IDevice {
    public:
        IDevice(uint32_t id);
        ~IDevice();
        uint32_t Id() const;
        bool GraphSubmit(vsi_nn_graph_t* graph, func_t func, data_t data);
        bool GraphRemove(const vsi_nn_graph_t* graph);
        bool ThreadExit();
        void WaitThreadIdle();

    protected:
        Device* device_;
};

}  // namespace vip

#endif
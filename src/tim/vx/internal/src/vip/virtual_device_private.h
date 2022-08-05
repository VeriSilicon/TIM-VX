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
#ifndef _VIP_VIRTUAL_DEVICE_PRIVATE_H
#define _VIP_VIRTUAL_DEVICE_PRIVATE_H

#include <memory>
#include <queue>
#include <vector>
#include <map>
#include <array>
#include <thread>
#include <iostream>
#include <mutex>
#include <unistd.h>
#include <condition_variable>
#include <functional>

extern "C" {
    #include "vsi_nn_pub.h"
};

namespace vip {

using func_t = std::function<bool (const void*)>;
using data_t = const void*;
typedef struct _Queueitem{
    size_t id;
    vsi_nn_graph_t* graph;
    func_t func;
    data_t data;
} QueueItem;

class GraphQueue{
    public:
        GraphQueue();
        ~GraphQueue(){};
        void Show();
        bool Submit(vsi_nn_graph_t* graph, func_t func, data_t data);
        bool Remove(const vsi_nn_graph_t* graph);
        QueueItem Fetch();
        bool Empty();
        size_t Size();
        void Notify();

    protected:
        std::vector<QueueItem> queue_;
        std::mutex queue_mtx_;
        std::condition_variable cv_;
        size_t gcount_;
};

class Worker{
    public:
        Worker();
        ~Worker(){};
        void Handle(const QueueItem& item);
        void RunGraph(const vsi_nn_graph_t* graph);
    protected:
};

class Device {
    public:
        Device(uint32_t id);
        ~Device();
        uint32_t Id() const;
        void ThreadInit();
        void StatusInit();
        bool ThreadExit();
        void HandleQueue();
        bool GraphSubmit(vsi_nn_graph_t* graph, func_t func, data_t data);
        bool GraphRemove(const vsi_nn_graph_t* graph);
        bool DeviceExit();
        bool ThreadIdle();
        void WaitThreadIdle();

    protected:
        uint32_t id_;
        std::array<std::thread, 2> threads_;
        std::unique_ptr<GraphQueue> graphqueue_;
        std::unique_ptr<Worker> worker_;
};

}  // namespace vip

#endif
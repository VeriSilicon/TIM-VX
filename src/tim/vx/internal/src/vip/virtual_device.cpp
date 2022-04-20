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
#include "vip/virtual_device.h"
#include "virtual_device_private.h"
#include "vsi_nn_log.h"

namespace vip {

Device::Device(uint32_t id) {
    id_ = id;
    graphqueue_ = std::make_unique<GraphQueue> ();
    worker_ = std::make_unique<Worker> ();;
    ThreadInit();
}

Device::~Device() {
    ThreadExit();
}

uint32_t Device::Id() const{
    return id_;
}

void Device::ThreadInit() {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
        std::thread t(&Device::HandleQueue, this);
        threads_[i] = std::move(t);
    }
}

bool Device::ThreadExit() {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
        graphqueue_->Submit(nullptr, NULL, NULL);  // submit fake graph to exit thread
    }
    for (std::size_t i = 0; i < threads_.size(); ++i) {
        if (threads_[i].joinable()) {
            threads_[i].join();
        }
    }
    return true;
}

bool Device::GraphSubmit(vsi_nn_graph_t* graph, func_t func, data_t data) {
    bool status = false;
    status = graphqueue_->Submit(graph, func, data);
    return status;
}

bool Device::GraphRemove(const vsi_nn_graph_t* graph) {
    return graphqueue_->Remove(graph);
}

void Device::WaitThreadIdle() {
    ThreadExit();
    ThreadInit();
}

Worker::Worker() {
}

void Worker::RunGraph(const vsi_nn_graph_t* graph) {
    vsi_nn_RunGraph(graph);
}

void Worker::Handle(const QueueItem& item) {
    vsi_nn_graph_t* graph = item.graph;
    func_t func = item.func;
    data_t data = item.data;
    size_t id = item.id;
    if (nullptr != graph) {
        VSILOGI("Start running graph%ld in thread[%ld] ", id , std::this_thread::get_id());
        RunGraph(graph);
        VSILOGI("End running graph%ld in thread[%ld]", id , std::this_thread::get_id());
    }
    if (NULL != func) {
        func(data);
    }
}

void Device::HandleQueue() {
    std::thread::id thd_id;
    thd_id = std::this_thread::get_id();
    while (1) {
        QueueItem item = graphqueue_->Fetch();
        if (0 == item.id) {  // exit when fetch fake graph
            // VSILOGD("Thread[%ld] exit", thd_id);
            break;
        }
        worker_->Handle(item);  // run graph
    }
}

GraphQueue::GraphQueue() {
    gcount_ = 1;  // 0 for fake graph
}

void GraphQueue::Show() {
    queue_mtx_.lock();
    VSILOGI("Queue element:");
    for (std::size_t i=0; i < queue_.size(); i++) {
        auto gid = queue_[i].id;
        VSILOGI("%d", gid);
    }
    queue_mtx_.unlock();
}

void GraphQueue::Notify() {
    cv_.notify_one();
}

bool GraphQueue::Submit(vsi_nn_graph_t* graph, func_t func, data_t data) {
    queue_mtx_.lock();
    QueueItem item;
    item.graph = graph;
    item.func = func;
    item.data = data;
    if (nullptr != graph) {
        item.id = gcount_;
        VSILOGI("Submit graph%ld", item.id);
        gcount_++;
        if (size_t(-1) == gcount_) {
            gcount_ = 1;
        }
    }
    else{
        item.id = 0;  // fake graph
    }
    queue_.push_back(item);
    queue_mtx_.unlock();
    Notify();
    return true;
}

QueueItem GraphQueue::Fetch() {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        QueueItem item = {(size_t)-1, nullptr, NULL, NULL};
        if (queue_.empty()) {
            cv_.wait(lock);
        }
        if (!queue_.empty()) {
            auto first = queue_[0];
            item.id = first.id;
            item.graph = first.graph;
            item.func = first.func;
            item.data = first.data;
            queue_.erase(queue_.begin());
        }
        // VSILOGD("Fetch graph%ld[%p] in thread[%ld]", item.id, item.graph, std::this_thread::get_id());
        return item;
}

bool GraphQueue::Remove(const vsi_nn_graph_t* graph) {
    queue_mtx_.lock();
    std::size_t idx=0;
    bool exist=false;
    if (!queue_.empty()) {
        for (std::size_t i=0; i < queue_.size(); i++) {
            if (graph == queue_[i].graph) {
                idx = i;
                exist = true;
            }
        }
        if (exist) {
            auto gid = queue_[idx].id;
            queue_.erase(queue_.begin() + idx);
            VSILOGI("Remove graph%ld", gid);
        }
    }
    queue_mtx_.unlock();
    return true;
}

bool GraphQueue::Empty() {
        queue_mtx_.lock();
        bool status = queue_.empty();
        queue_mtx_.unlock();
    return status;
}

size_t GraphQueue::Size() {
        queue_mtx_.lock();
        size_t size = queue_.size();
        queue_mtx_.unlock();
    return size;
}

IDevice::IDevice(uint32_t id) {
    device_ = new Device(id);
}

IDevice::~IDevice() {
    delete device_;
}

uint32_t IDevice::Id() const{
    return device_->Id();
}

bool IDevice::GraphSubmit(vsi_nn_graph_t* graph, func_t func, data_t data) {
    return device_->GraphSubmit(graph, func, data);
}

bool IDevice::GraphRemove(const vsi_nn_graph_t* graph) {
    return device_->GraphRemove(graph);
}

bool IDevice::ThreadExit() {
    return device_->ThreadExit();
}

void IDevice::WaitThreadIdle() {
    device_->WaitThreadIdle();
}

}  // namespace vip

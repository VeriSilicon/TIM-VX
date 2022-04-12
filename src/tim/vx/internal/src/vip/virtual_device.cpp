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

Device::Device(uint32_t id){
    id_ = id;
    graphqueue_ = std::make_unique<GraphQueue> ();
    worker_ = std::make_unique<Worker> ();;
    ThreadInit();
    StatusInit();
}

Device::~Device(){
}

uint32_t Device::Id() const{
    return id_;
}

void Device::ThreadInit(){
    for (std::size_t i = 0; i < threads_.size(); ++i){
        std::thread t(&Device::HandleQueue, this);
        threads_[i] = std::move(t);
    }
}

void Device::StatusInit(){
    // init thread status after thread id has been generated
    for (std::size_t i = 0; i < threads_.size(); ++i){
        VSILOGI("Init thread[%ld] status = %d", threads_[i].get_id(), IDLE);
        threads_status_[threads_[i].get_id()] = IDLE;
    }
}

bool Device::ThreadExit(){
    WaitThreadIdle();
    for (std::size_t i = 0; i < threads_.size(); ++i){
        threads_status_[threads_[i].get_id()] = CANCEL;
    }
    for (std::size_t i = 0; i < threads_.size(); ++i){
        graphqueue_->Submit(NULL, NULL, NULL);  // submit fake graph to exit thread
    }
    for (std::size_t i = 0; i < threads_.size(); ++i){
        threads_[i].join();
    }
    return true;
}

bool Device::GraphSubmit(vsi_nn_graph_t* graph, func_t func, data_t data){
    bool status = false;
    status = graphqueue_->Submit(graph, func, data);
    return status;
}

bool Device::GraphRemove(const vsi_nn_graph_t* graph){
    return graphqueue_->Remove(graph);
}

bool Device::ThreadIdle(){
    for (std::size_t i = 0; i < threads_.size(); ++i){
        if (threads_status_[threads_[i].get_id()] !=  IDLE){
            return false;
    }
  }
  return true;
}

void Device::WaitThreadIdle(){
  if (!ThreadIdle()){
    VSILOGI("Wait threads idle ...");
    std::unique_lock<std::mutex> lck(idle_mtx_);
    idle_cv_.wait(lck);
    VSILOGI("Threads idle");
  }
}

Worker::Worker(){
}

void Worker::RunGraph(const vsi_nn_graph_t* graph){
    vsi_nn_RunGraph(graph);
}

void Worker::Handle(const QueueItem& item){
    vsi_nn_graph_t* graph = item.graph;
    func_t func = item.func;
    data_t data = item.data;
    if (graph != NULL){
        VSILOGI("Start running graph%d in thread[%ld] ", item.id , std::this_thread::get_id());
        RunGraph(graph);
        VSILOGI("End running graph%d in thread[%ld]", item.id , std::this_thread::get_id());
    }
    if (func != NULL){
        func(data);
    }
}

void Device::HandleQueue(){
    std::thread::id thd_id;
    thd_id = std::this_thread::get_id();
    // VSILOGI("Thread[%ld] status = %d", thd_id, threads_status_[thd_id]);
    while (1) {
        QueueItem item = graphqueue_->Fetch();
        if (threads_status_[thd_id] == IDLE) {threads_status_[thd_id] = RUNNING;}
        worker_->Handle(item);
        if (threads_status_[thd_id] == RUNNING) {threads_status_[thd_id] = IDLE;}
        if (threads_status_[thd_id] == CANCEL) {VSILOGI("Thread[%ld] exit", thd_id); break;}
        if ((graphqueue_->Empty()) && ThreadIdle()) {idle_cv_.notify_one();}
    }
}

GraphQueue::GraphQueue(){
    gcount_ = 0;
}

void GraphQueue::Show(){
    queue_mtx_.lock();
    VSILOGI("Queue element:");
    for (std::size_t i=0; i < queue_.size(); i++){
        auto gid = queue_[i].id;
        VSILOGI("%d", gid);
    }
    queue_mtx_.unlock();
}

void GraphQueue::Notify(){
    cv_.notify_one();
}

bool GraphQueue::Submit(vsi_nn_graph_t* graph, func_t func, data_t data){
    queue_mtx_.lock();
    QueueItem item;
    item.graph = graph;
    item.func = func;
    item.data = data;
    item.id = gcount_;
    queue_.push_back(item);
    if (graph != NULL){
        VSILOGI("Submit graph%d", item.id);
        gcount_++;
    }
    queue_mtx_.unlock();
    Notify();
    return true;
}

QueueItem GraphQueue::Fetch(){
        QueueItem item;
        if (queue_.empty()){
            std::unique_lock<std::mutex> lock(queue_mtx_);
            cv_.wait(lock);
        }
        queue_mtx_.lock();
        if (!queue_.empty()){
            item = queue_.front();
            queue_.erase(queue_.begin());
        }
        queue_mtx_.unlock();
        return item;
}

bool GraphQueue::Remove(const vsi_nn_graph_t* graph){
    queue_mtx_.lock();
    std::size_t idx=0;
    bool exist=false;
    if (!queue_.empty()){
        for (std::size_t i=0; i < queue_.size(); i++){
            if (graph == queue_[i].graph){
                idx = i;
                exist = true;
            }
        }
        if (exist){
            auto gid = queue_[idx].id;
            queue_.erase(queue_.begin() + idx);
            VSILOGI("Remove graph%d", gid);
        }
    }
    queue_mtx_.unlock();
    return true;
}

bool GraphQueue::Empty() const{
    return queue_.empty();
}

IDevice::IDevice(uint32_t id){
    device_ = new Device(id);
}

IDevice::~IDevice(){
    delete device_;
}

uint32_t IDevice::Id() const{
    return device_->Id();
}

bool IDevice::GraphSubmit(vsi_nn_graph_t* graph, func_t func, data_t data){
    return device_->GraphSubmit(graph, func, data);
}

bool IDevice::GraphRemove(const vsi_nn_graph_t* graph){
    return device_->GraphRemove(graph);
}

bool IDevice::ThreadExit(){
    return device_->ThreadExit();
}

bool IDevice::ThreadIdle(){
    return device_->ThreadIdle();
}

void IDevice::WaitThreadIdle(){
    device_->WaitThreadIdle();
}

}  // namespace vip

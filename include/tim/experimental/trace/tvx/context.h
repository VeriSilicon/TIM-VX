/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
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
#ifndef TIM_EXPERIMENTAL_TRACE_TVX_CONTEXT_H_
#define TIM_EXPERIMENTAL_TRACE_TVX_CONTEXT_H_
#include "tim/vx/context.h"
#include "tim/experimental/trace/tvx/graph.h"
#include "tim/experimental/trace/tracer.h"

namespace trace {

namespace target = ::tim::vx;

struct Context : public TraceClassBase<target::Context> {
  DEF_INTERFACE_CONSTRUCTOR(Context)

  DEF_MEMFN_SP(Graph, CreateGraph)

  DEF_TRACED_API(bool, isClOnly)

  static inline std::shared_ptr<Context> Create() {
    std::string obj_name = Tracer::allocate_obj_name("ctx_");
    std::string pf(__PRETTY_FUNCTION__);
    pf.replace(pf.rfind("trace"), 5, target_namespace_name_);
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "auto %s =%s;\n", obj_name.c_str(),
            pf.substr(pf.rfind(" "), pf.size()).c_str());
    Tracer::logging_msg(log_msg);
    auto obj = std::make_shared<Context>(target::Context::Create());
    Tracer::insert_obj_name(static_cast<void*>(
        obj->TraceGetImplSp().get()), obj_name);
    return obj;
  }
};

} /* namespace trace */

#endif // TIM_EXPERIMENTAL_TRACE_TVX_CONTEXT_H_

/****************************************************************************
*
*    Copyright (c) 2022 Synaptics Corporation
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
#include "graph_synap.h"
#include <iostream>
#include "synap/ebg_utils.h"

using namespace std;

namespace tim {
namespace vx {


static string to_json(const vector<size_t>& sizes)
{
    string sep, s = "{";
    int i = 0;
    for (const auto& size: sizes) {
        s += sep + '"' + to_string(i++) + R"(":{"dtype":"byte","shape":[)" + to_string(size) + "]}";
        sep = ",";
    }
    return s + '}';
}


static string to_json(const vector<size_t>& inputs, const vector<size_t>& outputs)
{
    return "{\"Inputs\": " + to_json(inputs) + ",\"Outputs\": " + to_json(outputs) + "}";
}


static vector<size_t> tensor_sizes(const vector<shared_ptr<Tensor>>& tensors)
{
    vector<size_t> sizes;
    for (auto tensor: tensors) {
        sizes.push_back(tensor->GetSpec().GetByteSize());
    }
    return sizes;
}


bool GraphSynap::Compile() {
    VSILOGD("compile begin");
    size_t nbg_size = 0;
    if (!CompileToBinary(nullptr, &nbg_size)) {
        VSILOGE("Error getting NBG size");
        return false;
    }
    if (nbg_size == 0) {
        VSILOGE("Error NBG has size 0");
        return false;
    }
    VSILOGD("compile nbg_size: %zu", nbg_size);
    
    // generate binary graph does't require input data
    vector<uint8_t> nbg_buf(nbg_size);
    if (!CompileToBinary(nbg_buf.data(), &nbg_size)) {
        VSILOGE("Error compiling graph to NBG");
        return false;
    }
    VSILOGD("CompileToBinary done");
    
    uint8_t* ebg_buffer{};
    size_t ebg_size = nbg_to_ebg(nbg_buf.data(), nbg_size, &ebg_buffer, false);
    if (ebg_size == 0 || ebg_buffer == nullptr) {
        VSILOGE("NBG to EBG conversion failed");
        return false;
    }
    VSILOGD("NBG to EBG conversion done");

    vector<size_t> input_sizes = tensor_sizes(inputs_tensor_);
    vector<size_t> output_sizes = tensor_sizes(outputs_tensor_);
    if (!_network.load_model(ebg_buffer, ebg_size, to_json(input_sizes, output_sizes).c_str())) {
        VSILOGE("Error loading EBG model");
        return false;
    }

    return true;
}


bool GraphSynap::Run() {
    VSILOGD("GraphSynap::Run");
    // Copy data to network input tensors
    size_t ix = 0;
    for (auto& in: _network.inputs) {
        Tensor* t = inputs_tensor_[ix++].get();
        t->CopyDataFromTensor(in.data());
    }

    if (!_network.predict()) {
        VSILOGE("Error executing EBG model");
        return false;
    }

    // Copy data from network output tensors
    ix = 0;
    for (auto& out: _network.outputs) {
        Tensor* t = outputs_tensor_[ix++].get();
        t->CopyDataToTensor(out.data(), out.size());
    }
    
    return true;
}


}  // namespace vx
}  // namespace tim

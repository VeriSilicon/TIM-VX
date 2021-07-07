#include <iostream>

#include "tim/vx/graph.h"
#include "tim/vx/context.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/tensor.h"

#define ENABLE_RELU6 0

// TODO
template<typename T>
void fillRandomData(std::vector<T>& buf) {
    for(uint32_t i = 0; i < buf.size(); ++i) {
        buf[i] = i%static_cast<T>(255);
    }
}

enum ConfigField {
    kInImageW = 0,
    kInImageH = 1,
    kInImageChannel = 2,
    kKernelW = 3,
    kKernelH = 4,
    kOutChannel = 5,

    kConfigFiledCnt,

    kOutImageW = kInImageW,
    kOutImageH = kInImageH,
};

static const uint32_t default_cfg[] = {256, 256, 128, 3, 3, 1};

int main(int argc, char* argv[]) {

    bool single_layer_test = true;
    const uint32_t batch_sz = 1;
    uint32_t in_image_w = default_cfg[kInImageW];
    uint32_t in_image_h = default_cfg[kInImageH];
    uint32_t in_image_c = default_cfg[kInImageChannel];
    uint32_t kernel_w = default_cfg[kKernelW];
    uint32_t kernel_h = default_cfg[kKernelH];
    uint32_t out_image_w = default_cfg[kOutImageW];
    uint32_t out_image_h = default_cfg[kOutImageH];
    uint32_t out_image_c = default_cfg[kOutChannel];

    if (argc != 0 && argc != kConfigFiledCnt + 1){
        std::cout << "argc = " << argc << std::endl;
        std::cout << "Not enough parameter provided, will use default configuration" << std::endl;
    } else {
        argv ++;
        in_image_w = atoi(argv[kInImageW]);
        in_image_h = atoi(argv[kInImageH]);
        in_image_c = atoi(argv[kInImageChannel]);
        kernel_w = atoi(argv[kKernelW]);
        kernel_h = atoi(argv[kKernelH]);
        out_image_c = atoi(argv[kOutChannel]);
        out_image_h = atoi(argv[kOutImageH]);
        out_image_w = atoi(argv[kOutImageW]);
    }

    if (!single_layer_test && in_image_c != out_image_c) {
        std::cout << "Fatal error: multi-layer test only valid for ic = oc" << std::endl;
        return -1;
    }

    std::cout << "\n ===========================================================\n";
    std::cout << "\t test config: \n";
    #if ENABLE_RELU6
    std::cout << "\t Add activiation relu6 after convolution\n";
    #endif
    std::cout << "\t input image shape in (w,h,c): " << in_image_w << ", "
              << in_image_h << ", " << in_image_c << ", " << std::endl;
    std::cout << "\t kernel shape in (w, h, ic, oc): " << kernel_w << ", "
              << kernel_h << ", " << in_image_c << ", " << out_image_c << ", "
              << std::endl;
    std::cout << "\t output image shape in(w,h,c): " << out_image_w << ", " << out_image_h << ", " << out_image_c << ",\n";
    std::cout << " ===========================================================" << std::endl;

    tim::vx::ShapeType in_shape = {in_image_h, in_image_w, in_image_c, batch_sz};
    tim::vx::ShapeType kernel_shape = {kernel_w, kernel_h, in_image_c, out_image_c};
    tim::vx::ShapeType bias_shape = {out_image_c};
    tim::vx::ShapeType out_shape = {out_image_h, out_image_w, out_image_c, batch_sz};

    std::vector<uint8_t> in_data(batch_sz * in_image_w * in_image_h * in_image_c);
    fillRandomData(in_data);
    std::vector<uint8_t> kernel_data(kernel_w * kernel_h * in_image_c * out_image_c);
    fillRandomData(kernel_data);
    std::vector<uint32_t> bias_data(out_image_c);
    fillRandomData(bias_data);

    tim::vx::Quantization quant_type(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, in_shape, tim::vx::TensorAttribute::INPUT, quant_type);
    tim::vx::TensorSpec kernel_spec(tim::vx::DataType::UINT8, kernel_shape, tim::vx::TensorAttribute::CONSTANT, quant_type);
    tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape, tim::vx::TensorAttribute::CONSTANT, quant_type);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape, tim::vx::TensorAttribute::OUTPUT, quant_type);
    #if ENABLE_RELU6
    tim::vx::TensorSpec relu6_spec(tim::vx::DataType::UINT8, out_shape, tim::vx::TensorAttribute::TRANSIENT, quant_type);
    #endif

    auto context = tim::vx::Context::Create();
    auto graph = context->CreateGraph();

    auto input_tensor = graph->CreateTensor(input_spec, in_data.data());
    auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());
    auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
    #if ENABLE_RELU6
    auto relu6_input_tensor = graph->CreateTensor(relu6_spec);
    #endif

       auto output_tensor = graph->CreateTensor(output_spec);

    std::array<uint32_t, 4> pad = {
        (out_image_w - in_image_w + kernel_w - 1) / 2,
        (out_image_w - in_image_h + kernel_h - 1) / 2,
        (out_image_w - in_image_w + kernel_w - 1) / 2,
        (out_image_w - in_image_h + kernel_h - 1) / 2,
    };
    std::array<uint32_t, 2> stride = {1,1};
    std::array<uint32_t, 2> dilation = {1,1};
    auto conv2d_op = graph->CreateOperation<tim::vx::ops::Conv2d>(pad, stride, dilation);
    (*conv2d_op).BindInputs({input_tensor, kernel_tensor, bias_tensor});
    #if ENABLE_RELU6
    (*conv2d_op).BindOutput(relu6_input_tensor);
    #else
    if (single_layer_test) {
      (*conv2d_op).BindOutput(output_tensor);
    } else {  // multi-layer support
      tim::vx::TensorSpec temp_tensor_spec_0(
          tim::vx::DataType::UINT8, out_shape,
          tim::vx::TensorAttribute::TRANSIENT, quant_type);
      tim::vx::TensorSpec temp_tensor_spec_1(
          tim::vx::DataType::UINT8, out_shape,
          tim::vx::TensorAttribute::TRANSIENT, quant_type);

      auto temp_tensor_0 = graph->CreateTensor(temp_tensor_spec_0);
      auto temp_tensor_1 = graph->CreateTensor(temp_tensor_spec_1);

      auto kernel_0 = graph->CreateTensor(kernel_spec, kernel_data.data());
      auto kernel_1 = graph->CreateTensor(kernel_spec, kernel_data.data());
      auto bias_0 = graph->CreateTensor(bias_spec, bias_data.data());
      auto bias_1 = graph->CreateTensor(bias_spec, bias_data.data());

      (*conv2d_op).BindOutput(temp_tensor_0);
      auto conv2d_0 = graph->CreateOperation<tim::vx::ops::Conv2d>(pad, stride, dilation);
      (*conv2d_0).BindInputs({temp_tensor_0, kernel_0, bias_0}).BindOutput(temp_tensor_1);
      auto conv2d_1 = graph->CreateOperation<tim::vx::ops::Conv2d>(pad, stride, dilation);
      (*conv2d_1).BindInputs({temp_tensor_1, kernel_1, bias_1}).BindOutput(output_tensor);
    }
#endif

    #if ENABLE_RELU6
    auto relu6 = graph->CreateOperation<tim::vx::ops::Relu6>();
    (*relu6).BindInput(relu6_input_tensor).BindOutput(output_tensor);
    #endif

    graph->Compile();
    graph->Run();

    return 0;
}
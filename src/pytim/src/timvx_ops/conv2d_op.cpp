/***********************************
******  conv2d_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/conv2d.h"
#include "conv2d_op.h"


namespace TIMVXPY
{

    bool Conv2dCreator::parse_weights(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_value<py::int_, uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    bool Conv2dCreator::parse_padding(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_pad_type(op_info, m_op_name, "padding", op_attr.padding, false);
    }

    bool Conv2dCreator::parse_ksize(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize, false);
    }

    bool Conv2dCreator::parse_stride(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Conv2dCreator::parse_dilation(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "dilation", op_attr.dilation);
    }

    bool Conv2dCreator::parse_pad(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad, false);
    }

    bool Conv2dCreator::parse_multiplier(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_value<py::int_, int32_t>(op_info, m_op_name, "multiplier", op_attr.multiplier, false);
    }

    bool Conv2dCreator::parse_input_layout(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_data_layout_type(op_info, m_op_name, "input_layout", op_attr.input_layout, false);
    }
    
    bool Conv2dCreator::parse_kernel_layout(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        return parse_data_layout_type(op_info, m_op_name, "kernel_layout", op_attr.input_layout, false);
    }

    bool Conv2dCreator::parse_op_attr(const py::dict &op_info, Conv2dOpAttr &op_attr)
    {
        op_attr.weights = 0;
        op_attr.padding = PadType::AUTO;
        op_attr.ksize = {0, 0};
        op_attr.multiplier = 0;
        op_attr.pad = {0, 0, 0, 0};
        op_attr.input_layout = DataLayout::WHCN;
        op_attr.kernel_layout = DataLayout::WHIcOc;
        return parse_weights(op_info, op_attr) && parse_padding(op_info, op_attr)
            && parse_ksize(op_info, op_attr) && parse_stride(op_info, op_attr)
            && parse_dilation(op_info, op_attr) && parse_pad(op_info, op_attr)
            && parse_multiplier(op_info, op_attr) && parse_input_layout(op_info, op_attr)
            && parse_kernel_layout(op_info, op_attr);
    }

    Operation* Conv2dCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        Conv2dOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        uint32_t                weights        = op_attr.weights;
        PadType                 padding        = op_attr.padding;
        std::array<uint32_t, 2> ksize          = op_attr.ksize;
        std::array<uint32_t, 2> stride         = op_attr.stride;
        std::array<uint32_t, 2> dilation       = op_attr.dilation;
        std::array<uint32_t, 4> pad            = op_attr.pad;
        int32_t                 multiplier     = op_attr.multiplier;
        DataLayout              input_layout   = op_attr.input_layout;
        DataLayout              kernel_layout  = op_attr.kernel_layout;
        // std::cout << "weights: " << weights << std::endl;
        // std::cout << "padding: " << (int)padding << std::endl;
        // std::cout << "ksize: " << ksize[0] << " " << ksize[1] << std::endl;
        // std::cout << "stride: " << stride[0] << " " << stride[1] << std::endl;
        // std::cout << "dilation: " << dilation[0] << " " << dilation[1] << std::endl;
        // std::cout << "pad: " << pad[0] << " " << pad[1] << " " << pad[2] << " " << pad[3] << std::endl;
        // std::cout << "multiplier: " << multiplier << std::endl;
        // std::cout << "input_layout: " << (int)input_layout << std::endl;
        // std::cout << "kernel_layout: " << (int)kernel_layout << std::endl;
        return graph->CreateOperation<ops::Conv2d>(weights, padding, ksize, stride, 
            dilation, pad, multiplier, input_layout, kernel_layout).get();
    }

    REGISTER_OP_CREATOR(Conv2dCreator, Conv2d);
} // namespace TIMVXPY
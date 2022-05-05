/***********************************
******  conv2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{
    
    class Conv2dCreator : public OpCreator
    {
    public:
        struct Conv2dOpAttr
        {
            uint32_t                weights;
            PadType                 padding;
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            std::array<uint32_t, 2> dilation;
            std::array<uint32_t, 4> pad;
            int32_t                 multiplier;
            DataLayout              input_layout;
            DataLayout              kernel_layout;
        };

        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool get_conv2d_type();
        bool parse_weights(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_padding(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_ksize(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_stride(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_dilation(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_pad(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_multiplier(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_input_layout(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_kernel_layout(const py::dict &op_info, Conv2dOpAttr &op_attr);
        bool parse_op_attr(const py::dict &op_info, Conv2dOpAttr &op_attr);

    private:
        std::string m_op_name = "Conv2d";
    };

} // namespace TIMVXPY

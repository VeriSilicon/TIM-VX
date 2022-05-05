/***********************************
******  resize_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{

    class ResizeCreator : public OpCreator
    {
    public:
        struct ResizeOpAttr
        {
            ResizeType type;
            float factor;
            bool align_corners;
            bool half_pixel_centers;
            int target_height;
            int target_width;
            DataLayout layout;
        };

        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool parse_op_attr(const py::dict &op_info, ResizeOpAttr &op_attr);

    private:
        std::string m_op_name = "Resize";
    };

} // namespace TIMVXPY

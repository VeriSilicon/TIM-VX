/***********************************
******  resize_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/resize.h"
#include "resize_op.h"


namespace TIMVXPY
{
    bool ResizeCreator::parse_op_attr(const py::dict &op_info, ResizeOpAttr &op_attr)
    {
        op_attr.layout = DataLayout::WHCN;
        return parse_resize_type(op_info, m_op_name, "type", op_attr.type)
            && parse_value<py::float_, float>(op_info, m_op_name, "factor", op_attr.factor)
            && parse_value<py::bool_, bool>(op_info, m_op_name, "align_corners", op_attr.align_corners)
            && parse_value<py::bool_, bool>(op_info, m_op_name, "half_pixel_centers", op_attr.half_pixel_centers)
            && parse_value<py::int_, int32_t>(op_info, m_op_name, "target_height", op_attr.target_height)
            && parse_value<py::int_, int32_t>(op_info, m_op_name, "target_width", op_attr.target_width)
            && parse_data_layout_type(op_info, m_op_name, "layout", op_attr.layout);
    }

    Operation* ResizeCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        ResizeOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        ResizeType type         = op_attr.type;
        float factor            = op_attr.factor;
        bool align_corners      = op_attr.align_corners;
        bool half_pixel_centers = op_attr.half_pixel_centers;
        int target_height       = op_attr.target_height;
        int target_width        = op_attr.target_width;
        DataLayout layout       = op_attr.layout;
        return graph->CreateOperation<ops::Resize>(type, factor, align_corners,
            half_pixel_centers, target_height, target_width, layout).get();
    }

    REGISTER_OP_CREATOR(ResizeCreator, Resize);
} // namespace TIMVXPY
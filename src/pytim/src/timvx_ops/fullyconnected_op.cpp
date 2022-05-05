/***********************************
******  fullconnected_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/fullyconnected.h"
#include "fullyconnected_op.h"


namespace TIMVXPY
{
    bool FullyConnectedCreator::parse_op_attr(const py::dict &op_info, FullyConnectedOpAttr &op_attr)
    {
        op_attr.weights = 0;
        return parse_value<py::int_, uint32_t>(op_info, m_op_name, "axis", op_attr.axis) &&
            parse_value<py::int_, uint32_t>(op_info, m_op_name, "weights", op_attr.weights, false);
    }

    Operation* FullyConnectedCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        FullyConnectedOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        uint32_t axis    = op_attr.axis;
        uint32_t weights = op_attr.weights;
        return graph->CreateOperation<ops::FullyConnected>(axis, weights).get();
    }

    REGISTER_OP_CREATOR(FullyConnectedCreator, FullyConnected);
} // namespace TIMVXPY
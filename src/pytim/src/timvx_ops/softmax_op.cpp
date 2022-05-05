/***********************************
******  softmax_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/softmax.h"
#include "softmax_op.h"

namespace TIMVXPY
{
    bool SoftmaxCreator::parse_op_attr(const py::dict &op_info, SoftmaxOpAttr &op_attr)
    {
        return parse_value<py::float_, float>(op_info, m_op_name, "beta", op_attr.beta) &&
            parse_value<py::int_, int32_t>(op_info, m_op_name, "axis", op_attr.axis);
    }

    Operation* SoftmaxCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        SoftmaxOpAttr op_attr;
        if (!parse_op_attr(op_info, op_attr))
            return nullptr;

        float beta   = op_attr.beta;
        int32_t axis = op_attr.axis;
        return graph->CreateOperation<ops::Softmax>(beta, axis).get();
    }

    REGISTER_OP_CREATOR(SoftmaxCreator, Softmax);
} // namespace TIMVXPY
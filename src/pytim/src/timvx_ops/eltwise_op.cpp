/***********************************
******  eltwise_op.cpp
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#include "tim/vx/ops/elementwise.h"
#include "eltwise_op.h"


namespace TIMVXPY
{
    Operation* EltwiseCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        std::string eltwise_type;
        if (!parse_value<py::str, std::string>(op_info, m_op_name, "eltwise_type", eltwise_type))
            return nullptr;
        std::string eltwise_op_name = m_op_name + "_" + eltwise_type;
        if ("Minimum" == eltwise_type)
        {
            return graph->CreateOperation<ops::Minimum>().get();
        }
        else if ("Maximum" == eltwise_type)
        {
            return graph->CreateOperation<ops::Maximum>().get();
        }
        else if ("Add" == eltwise_type)
        {
            return graph->CreateOperation<ops::Add>().get();
        }
        else if ("Sub" == eltwise_type)
        {
            return graph->CreateOperation<ops::Sub>().get();
        }
        else if ("Pow" == eltwise_type)
        {
            return graph->CreateOperation<ops::Pow>().get();
        }
        else if ("FloorDiv" == eltwise_type)
        {
            return graph->CreateOperation<ops::FloorDiv>().get();
        }
        else if ("Multiply" == eltwise_type || "Div" == eltwise_type)
        {
            float scale = 1.0f;
            if (!parse_value<py::float_, float>(op_info, eltwise_op_name, "scale", scale, false))
                return nullptr;
            if ("Multiply" == eltwise_type)
                return graph->CreateOperation<ops::Multiply>(scale).get();
            else
                return graph->CreateOperation<ops::Div>(scale).get();
        }
        else
            std::cout << "unsupported elewise op type: " << eltwise_type << std::endl;
        return nullptr;
    }

    REGISTER_OP_CREATOR(EltwiseCreator, Eltwise);
} // namespace TIMVXPY
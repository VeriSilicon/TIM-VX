/***********************************
******  activation_op.cpp
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#include "tim/vx/ops/activations.h"
#include "activation_op.h"

namespace TIMVXPY
{

    bool ActivationCreator::parse_prelu_attr(const py::dict &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_prelu";
        return parse_value<py::int_, int>(op_info, full_op_name, "axis", op_attr.prelu.axis);
    }

    bool ActivationCreator::parse_leakyrelu_attr(const py::dict &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_leakyrelu";
        return parse_value<py::float_, float>(op_info, full_op_name, "ratio", op_attr.leakyrelu.ratio);
    }

    bool ActivationCreator::parse_linear_attr(const py::dict &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_linear";
        return parse_value<py::float_, float>(op_info, full_op_name, "a", op_attr.linear.a)
            && parse_value<py::float_, float>(op_info, full_op_name, "b", op_attr.linear.b, false);
    }

    bool ActivationCreator::parse_gelu_attr(const py::dict &op_info, ActivationOpAttr &op_attr)
    {
        std::string full_op_name = m_op_name + "_gelu";
        return parse_value<py::bool_, bool>(op_info, full_op_name, "approximate", op_attr.gelu.approximate, false);
    }

    bool ActivationCreator::parse_op_attr(std::string op_type, const py::dict &op_info, ActivationOpAttr &op_attr)
    {
        op_attr.gelu.approximate = true;
        op_attr.linear.b = 0.0f;
        if ("prelu" == op_type)
            return parse_prelu_attr(op_info, op_attr);
        else if ("leakyrelu" == op_type)
            return parse_leakyrelu_attr(op_info, op_attr);
        else if ("linear" == op_type)
            return parse_linear_attr(op_info, op_attr);
        else if ("gelu" == op_type)
            return parse_gelu_attr(op_info, op_attr);
        else
            return true;
    }

    Operation* ActivationCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        ActivationOpAttr op_attr;
        std::string activation_type;
        if (!parse_value<py::str, std::string>(op_info, m_op_name, "activation_type", activation_type))
            return nullptr;
        if (!parse_op_attr(activation_type, op_info, op_attr))
            return nullptr;
        if ("Relu" == activation_type)
        {
            return graph->CreateOperation<ops::Relu>().get();
        }
        else if ("Relu1" == activation_type)
        {
            return graph->CreateOperation<ops::Relu1>().get();
        }
        else if ("Relu6" == activation_type)
        {
            return graph->CreateOperation<ops::Relu6>().get();
        }
        else if ("Elu" == activation_type)
        {
            return graph->CreateOperation<ops::Elu>().get();
        }
        else if ("Sigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::Sigmoid>().get();
        }
        else if ("Mish" == activation_type)
        {
            return graph->CreateOperation<ops::Mish>().get();
        }
        else if ("HardSigmoid" == activation_type)
        {
            return graph->CreateOperation<ops::HardSigmoid>().get();
        }
        else if ("SoftRelu" == activation_type)
        {
            return graph->CreateOperation<ops::SoftRelu>().get();
        }
        else if ("HardSwish" == activation_type)
        {
            return graph->CreateOperation<ops::HardSwish>().get();
        }
        else if ("Swish" == activation_type)
        {
            return graph->CreateOperation<ops::Swish>().get();
        }
        else if ("Prelu" == activation_type)
        {
            int axis = op_attr.prelu.axis;
            return graph->CreateOperation<ops::Prelu>(axis).get();
        }        
        else if ("Tanh" == activation_type)
        {
            return graph->CreateOperation<ops::Tanh>().get();
        }
        else if ("LeakyRelu" == activation_type)
        {
            float ratio = op_attr.leakyrelu.ratio;
            return graph->CreateOperation<ops::LeakyRelu>(ratio).get();
        }
        else if ("Linear" == activation_type)
        {
            float a = op_attr.linear.a;
            float b = op_attr.linear.b;
            return graph->CreateOperation<ops::Linear>(a, b).get();
        }
        else if ("Gelu" == activation_type)
        {
            bool approximate = op_attr.gelu.approximate;
            return graph->CreateOperation<ops::Gelu>(approximate).get();
        }
        else
            std::cout << "unsupported activation op type: " << activation_type << std::endl; 
        return nullptr;
    }

    REGISTER_OP_CREATOR(ActivationCreator, Activation);
} // namespace TIMVXPY
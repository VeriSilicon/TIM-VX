/***********************************
******  activation_op.h
******
******  Created by zhaojd on 2022/04/29.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{

    class ActivationCreator : public OpCreator
    {
    public:
        struct ActivationOpAttr
        {
            // prelu parameter
            struct
            {
                int axis;
            }prelu;
            // leakyrelu parameter
            struct
            {
                float ratio = 1.0f;
            }leakyrelu;        
            // linear parameter
            struct
            {
                float a = 1.0f;
                float b = 0.0f;
            }linear;
            // gelu parameter
            struct
            {
                bool approximate = true;
            }gelu;
            // hard sigmoid parameter
            struct
            {
                float alpha;
                float beta;
            }hardsigmoid;
        };

        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool parse_prelu_attr(const py::dict &op_info, ActivationOpAttr &op_attr);
        bool parse_leakyrelu_attr(const py::dict &op_info, ActivationOpAttr &op_attr);
        bool parse_linear_attr(const py::dict &op_info, ActivationOpAttr &op_attr);
        bool parse_gelu_attr(const py::dict &op_info, ActivationOpAttr &op_attr);
        bool parse_hardsigmoid_attr(const py::dict &op_info, ActivationOpAttr &op_attr);
        bool parse_op_attr(std::string op_type, const py::dict &op_info, ActivationOpAttr &op_attr);
    
    private:
        std::string m_op_name = "Activation";
    };

} // namespace TIMVXPY

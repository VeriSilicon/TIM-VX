/***********************************
******  softmax_op.h
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{

    class SoftmaxCreator : public OpCreator
    {
    public:
        struct SoftmaxOpAttr
        {
            float beta;
            int32_t axis;
        };
    
        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool parse_op_attr(const py::dict &op_info, SoftmaxOpAttr &op_attr);

    private:
        std::string m_op_name = "Softmax";
    };
} // namespace TIMVXPY

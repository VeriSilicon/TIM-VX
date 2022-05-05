/***********************************
******  eltwise_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{

    class EltwiseCreator : public OpCreator
    {
    public:
        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        std::string m_op_name = "Eltwise";
    };

} // namespace TIMVXPY

/***********************************
******  concat_op.h
******
******  Created by zhaojd on 2022/05/11.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{

    class ConcatCreator : public OpCreator
    {
    public:
        struct ConcatOpAttr
        {
            uint32_t axis;
            int input_cnt;
        };
    
        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        bool parse_op_attr(const py::dict &op_info, ConcatOpAttr &op_attr);

    private:
        std::string m_op_name = "Concat";
    };
} // namespace TIMVXPY

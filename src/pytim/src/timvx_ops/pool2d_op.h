/***********************************
******  pool2d_op.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include "op_creator.h"
using namespace tim::vx;
using namespace std;
namespace TIMVXPY
{
    class Pool2dCreator : public OpCreator
    {
    public:
        struct Pool2dOpAttr
        {
            // pool2d common
            PoolType type;
            RoundType round_type;
            DataLayout layout;
            // classic pool2d common
            std::array<uint32_t, 2> ksize;
            std::array<uint32_t, 2> stride;
            // Classic Pool2d 1
            PadType padding;
            // Classic Pool2d 2
            std::array<uint32_t, 4> pad;
            // global and adaptive pool2d
            std::array<uint32_t, 2> input_size;
            // adaptive pool2d
            std::array<uint32_t, 2> output_size;
        };

        enum Pool2dCfgType
        {
            None,
            Classic_Pool2d_1,
            Classic_Pool2d_2,
            Global_Pool2d,
            Adaptive_Pool2d,
        };

        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) override;

    private:
        Pool2dCfgType get_pool2d_type(const py::dict &op_info);
        bool parse_pad(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_padding(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_type(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_ksize(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_stride(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_input_size(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_output_size(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_round_type(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_layout(const py::dict &op_info, Pool2dOpAttr &op_attr);
        bool parse_op_attr(const py::dict &op_info, Pool2dOpAttr &op_attr, Pool2dCfgType pool_type);

    private:
        std::string m_op_name = "Pool2d";
    };

} // namespace TIMVXPY

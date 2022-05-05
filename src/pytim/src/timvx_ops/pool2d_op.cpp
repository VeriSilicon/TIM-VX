/***********************************
******  pool2d_op.cpp
******
******  Created by zhaojd on 2022/05/02.
***********************************/
#include "tim/vx/ops/pool2d.h"
#include "pool2d_op.h"


namespace TIMVXPY
{
    Pool2dCreator::Pool2dCfgType Pool2dCreator::get_pool2d_type(const py::dict &op_info)
    {
        if (op_info.contains("padding") && op_info.contains("pad"))
        {
            std::cout << "cannot contain padding and pad same time!" << std::endl;
            return Pool2dCfgType::None;
        }
        if ((op_info.contains("padding") || op_info.contains("pad"))
            && op_info.contains("input_size"))
        {
            std::cout << "cannot contain padding(pad) and input_size same time!" << std::endl;
            return Pool2dCfgType::None;
        }
        if (op_info.contains("type") && op_info.contains("padding") &&
            op_info.contains("ksize") && op_info.contains("stride"))
            return Pool2dCfgType::Classic_Pool2d_1;
        else if (op_info.contains("type") && op_info.contains("pad") &&
            op_info.contains("ksize") && op_info.contains("stride"))
            return Pool2dCfgType::Classic_Pool2d_2;
        else if (op_info.contains("type") && op_info.contains("input_size") &&
            !op_info.contains("ksize"))
            return Pool2dCfgType::Global_Pool2d;
        else if (op_info.contains("type") && op_info.contains("input_size") &&
            op_info.contains("output_size"))
            return Pool2dCfgType::Adaptive_Pool2d;
        else
        {
            std::cout << "invalid pool2d op attr!" << std::endl;
            return Pool2dCfgType::None;
        }
    }

    bool Pool2dCreator::parse_type(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_pool_type(op_info, m_op_name, "type", op_attr.type);
    }

    bool Pool2dCreator::parse_padding(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_pad_type(op_info, m_op_name, "padding", op_attr.padding);
    }

    bool Pool2dCreator::parse_pad(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 4>(op_info, m_op_name, "pad", op_attr.pad);
    }

    bool Pool2dCreator::parse_ksize(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "ksize", op_attr.ksize);
    }

    bool Pool2dCreator::parse_stride(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "stride", op_attr.stride);
    }

    bool Pool2dCreator::parse_input_size(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "input_size", op_attr.input_size);
    }

    bool Pool2dCreator::parse_output_size(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_fix_list<py::int_, uint32_t, 2>(op_info, m_op_name, "output_size", op_attr.output_size);
    }

    bool Pool2dCreator::parse_round_type(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return OpCreator::parse_round_type(op_info, m_op_name, "round_type", op_attr.round_type, false);
    }

    bool Pool2dCreator::parse_layout(const py::dict &op_info, Pool2dOpAttr &op_attr)
    {
        return parse_data_layout_type(op_info, m_op_name, "layout", op_attr.layout, false);
    }

    bool Pool2dCreator::parse_op_attr(const py::dict &op_info, Pool2dOpAttr &op_attr, Pool2dCfgType cfg_type)
    {
        op_attr.round_type = RoundType::FLOOR;
        op_attr.layout = DataLayout::WHCN;
        if (Classic_Pool2d_1 == cfg_type)
            return parse_type(op_info, op_attr) && parse_padding(op_info, op_attr)
                && parse_ksize(op_info, op_attr) && parse_stride(op_info, op_attr)
                && parse_round_type(op_info, op_attr) && parse_layout(op_info, op_attr);

        if (Classic_Pool2d_2 == cfg_type)
            return parse_type(op_info, op_attr) && parse_pad(op_info, op_attr)
                && parse_ksize(op_info, op_attr) && parse_stride(op_info, op_attr)
                && parse_round_type(op_info, op_attr) && parse_layout(op_info, op_attr);

        if (Global_Pool2d == cfg_type)
            return parse_type(op_info, op_attr) && parse_input_size(op_info, op_attr)
                && parse_round_type(op_info, op_attr) && parse_layout(op_info, op_attr);

        if (Adaptive_Pool2d == cfg_type)
            return parse_type(op_info, op_attr) && parse_input_size(op_info, op_attr)
                && parse_output_size(op_info, op_attr) && parse_round_type(op_info, op_attr) 
                && parse_layout(op_info, op_attr);
    }

    Operation* Pool2dCreator::on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info)
    {
        std::map<Pool2dCfgType, std::string> pool_cfg_type_map;
        pool_cfg_type_map[Classic_Pool2d_1] = "Classic_Pool2d_1";
        pool_cfg_type_map[Classic_Pool2d_2] = "Classic_Pool2d_2";
        pool_cfg_type_map[Global_Pool2d] = "Global_Pool2d";
        pool_cfg_type_map[Adaptive_Pool2d] = "Adaptive_Pool2d";
        Pool2dCfgType cfg_type;
        Pool2dOpAttr op_attr;
        cfg_type = get_pool2d_type(op_info);
        if (pool_cfg_type_map.find(cfg_type) == pool_cfg_type_map.end())
        {
            std::cout << "unsupported pool cfg, please check!" << std::endl;
            return nullptr;
        }
        if (!parse_op_attr(op_info, op_attr, cfg_type))
            return nullptr;

        PoolType type                       = op_attr.type;
        PadType padding                     = op_attr.padding;
        std::array<uint32_t, 2> ksize       = op_attr.ksize;
        std::array<uint32_t, 2> stride      = op_attr.stride;
        RoundType round_type                = op_attr.round_type;
        DataLayout layout                   = op_attr.layout;
        std::array<uint32_t, 4> pad         = op_attr.pad;
        std::array<uint32_t, 2> input_size  = op_attr.input_size;
        std::array<uint32_t, 2> output_size = op_attr.output_size;
        switch (cfg_type)
        {
            case Classic_Pool2d_1:
                return graph->CreateOperation<ops::Pool2d>(type, padding,
                    ksize, stride, round_type, layout).get();
            case Classic_Pool2d_2:
                return graph->CreateOperation<ops::Pool2d>(type, pad,
                    ksize, stride, round_type, layout).get();
            case Global_Pool2d:
                return graph->CreateOperation<ops::Pool2d>(type, input_size,
                    round_type, layout).get();
            case Adaptive_Pool2d:
                return graph->CreateOperation<ops::Pool2d>(type, input_size,
                    output_size, round_type, layout).get();
            default:
                std::cout << "get invalid pool2d type!" << std::endl;
                return nullptr;
        }
    }

    REGISTER_OP_CREATOR(Pool2dCreator, Pool2d);
} // namespace TIMVXPY
/***********************************
******  timvx_engine.cpp
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#include <mutex>
#include <iostream>
#include "timvx_engine.h"
#include "tensor_info.h"
#include "timvx_ops/op_creator.h"

namespace TIMVXPY
{
    extern void register_ops();
    #define BITS_PER_BYTE 8
    TimVXEngine::TimVXEngine(const std::string &graph_name)
    {
        m_context.reset();
        m_graph.reset();
        m_graph_name = graph_name;
        static std::once_flag flag;
        std::call_once(flag, &register_ops);
    }

    TimVXEngine::~TimVXEngine()
    {
        m_tensors_data.clear();
        m_tensors.clear();
        m_graph.reset();
        m_context.reset();
    }

    uint32_t TimVXEngine::type_get_bits(DataType type)
    {   
        switch( type )
        {
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::BOOL8:
                return 8;
            case DataType::INT16:
            case DataType::UINT16:
            case DataType::FLOAT16:
                return 16;
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::FLOAT32:
                return 32;
            default:
                return 0;
        }
    }  

    size_t TimVXEngine::get_tensor_size(const std::string &tensor_name)
    {
        size_t sz;
        size_t i;
        size_t bits_num;
        size_t dim_num;
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            std::cout << "tensor " << tensor_name <<" not exists!" << std::endl;
            return 0;
        }
        sz = 0;
        auto tensor = m_tensors[tensor_name];
        dim_num = tensor->GetShape().size();
        auto shape = tensor->GetShape();
        auto type = tensor->GetDataType();
        if(0 == dim_num)
        {
            return sz;
        }
        bits_num = type_get_bits( type );
        if( bits_num < BITS_PER_BYTE )
        {
            if(shape[0] % 2 == 0)
            {
                sz = shape[0] / 2;
            }
            else
            {
                sz = shape[0] / 2 + shape[0] % 2;
            }
        }
        else
        {
            sz = shape[0] * bits_num / BITS_PER_BYTE;
        }
        for( i = 1; i < dim_num; i ++ )
        {
            sz *= shape[i];
        }
        return sz;
    }

    bool TimVXEngine::create_tensor(const std::string &tensor_name, const py::dict &tensor_info)
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        if (m_tensors.find(tensor_name) != m_tensors.end())
        {
            std::cout << "duplicate tensor name is provided, please check again!" << std::endl;
            return false;
        }
        TensorSpec tensor_spec;
        if (!TensorSpecConstruct::construct_tensorspec(tensor_info, tensor_name, tensor_spec))
        {
            std::cout << "construct tensor spec fail, please check again!" << std::endl;
            return false;
        }
        std::shared_ptr<Tensor> tensor;
        if (!tensor_info.contains("data"))
            tensor = m_graph->CreateTensor(tensor_spec);
        else
        {
            py::array data_array = py::cast<py::array>(tensor_info["data"]);
            int num_bytes = data_array.nbytes();
            std::shared_ptr<char> data_array_ptr(new char[num_bytes], [](char* data_array_ptr){delete [] data_array_ptr;});
            m_tensors_data[tensor_name] = data_array_ptr;
            memcpy((void*)data_array_ptr.get(), data_array.data(), num_bytes);
            tensor = m_graph->CreateTensor(tensor_spec, (void*)data_array_ptr.get());
        }
        if (nullptr == tensor.get())
        {
            std::cout << "graph create tensor " << tensor_name <<" fail!" << std::endl;
            return false;
        }
        m_tensors[tensor_name] = tensor;
        return true;
    }    

    bool TimVXEngine::copy_data_from_tensor(const std::string &tensor_name, py::buffer &np_data)
    {
        py::buffer_info buf = np_data.request();
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            std::cout << "tensor " << tensor_name <<" not exists!" << std::endl;
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        if (buf.ndim != tensor->GetShape().size())
        {
            std::cout << "tensor dim:" << tensor->GetShape().size() << " not equal to numpy data dim:" << 
                buf.ndim << std::endl;
            return false;
        }
        size_t total_np_size = buf.size * buf.itemsize;
        size_t total_tensor_size = get_tensor_size(tensor_name);
        if (total_tensor_size != total_np_size)
        {
            std::cout << "tensor size:" << total_tensor_size << " not equal to numpy data size:" << 
                total_np_size << std::endl;
            return false;
        }
        if (total_tensor_size <= 0)
        {
            std::cout << "tensor size:" << total_tensor_size << " not valid!" << std::endl;
            return false;
        }
        return tensor->CopyDataFromTensor(buf.ptr);
    }

    bool TimVXEngine::copy_data_to_tensor(const std::string &tensor_name, py::buffer &np_data)
    {
        py::buffer_info buf = np_data.request();
        if (m_tensors.find(tensor_name) == m_tensors.end())
        {
            std::cout << "tensor " << tensor_name <<" not exists!" << std::endl;
            return false;
        }
        auto tensor = m_tensors[tensor_name];
        // if (buf.ndim != tensor->GetShape().size())
        // {
        //     std::cout << "tensor dim:" << tensor->GetShape().size() << " not equal to numpy data dim:" << 
        //         buf.ndim << std::endl;
        //     return false;
        // }
        int total_np_size = buf.size * buf.itemsize;
        int total_tensor_size = get_tensor_size(tensor_name);
        if (total_tensor_size != total_np_size)
        {
            std::cout << "tensor size:" << total_tensor_size << " not equal to numpy data size:" << 
                total_np_size << std::endl;
            return false;
        }
        if (total_tensor_size <= 0)
        {
            std::cout << "tensor size:" << total_tensor_size << " not valid!" << std::endl;
            return false;
        }
        return tensor->CopyDataToTensor(buf.ptr, total_np_size);
    }

    bool TimVXEngine::create_operation(py::dict &op_info)
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        if (!op_info.contains("op_type") || !py::isinstance<py::str>(op_info["op_type"]))
        {
            std::cout << "op_type item is not contained, or op_type is not string!" << std::endl;
            return false;
        }
        if (!op_info.contains("op_name") || !py::isinstance<py::str>(op_info["op_name"]))
        {
            std::cout << "op_name item is not contained, or op_name is not string!" << std::endl;
            return false;
        }
        if (!op_info.contains("op_attr") || !py::isinstance<py::dict>(op_info["op_attr"]))
        {
            std::cout << "op_attr item is not contained, or op_attr is not dict!" << std::endl;
            return false;
        }
        if (op_info.contains("rounding_policy") && !py::isinstance<py::dict>(op_info["rounding_policy"]))
        {
            std::cout << "rounding_policy item is contained, but not dict!" << std::endl;
            return false;
        }
        auto op_name = std::string(py::str(op_info["op_name"]));
        if (m_operations.find(op_name) != m_operations.end())
        {
            std::cout << op_name << " is duplicate!" << std::endl;
            return false;
        }
        auto op_type = std::string(py::str(op_info["op_type"]));
        OpCreator* op_creator = TimVXOp::get_instance()->get_creator(op_type);
        if (nullptr == op_creator)
        {
            std::cout << op_type << " op creator not find!" << std::endl;
            return false;
        }
        auto op_node = op_creator->on_create(m_graph, py::dict(op_info["op_attr"]));
        if (nullptr != op_node && op_info.contains("rounding_policy"))
        {
            py::dict rounding_policy = py::dict(op_info["rounding_policy"]);
            OverflowPolicy overflow_policy_type = OverflowPolicy::SATURATE;
            RoundingPolicy rounding_policy_type = RoundingPolicy::RTNE;
            RoundType      round_type           = RoundType::FLOOR;
            uint32_t       accumulator_bits     = 0;
            op_creator->parse_overflow_policy_type(rounding_policy, op_name, "overflow_policy", overflow_policy_type, false);
            op_creator->parse_rounding_policy_type(rounding_policy, op_name, "rounding_policy", rounding_policy_type, false);
            op_creator->parse_round_type(rounding_policy, op_name, "down_scale_size_rounding", round_type, false);
            op_creator->parse_value<py::int_, uint>(rounding_policy, op_name, "accumulator_bits", accumulator_bits, false);
            op_node->SetRoundingPolicy(overflow_policy_type, rounding_policy_type, round_type, accumulator_bits);
        }
        if (nullptr != op_node)
        {
            m_operations[op_name] = op_node;
            m_op_info[op_name] = op_info;
            return true;
        }        
        return false;
    }

    bool TimVXEngine::bind_inputs(const std::string &op_name, const std::vector<std::string> &input_list)
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            std::cout << "op " << op_name <<" not exists!" << std::endl;
            return false;
        }
        if (input_list.size() <= 0)
        {
            std::cout << "bind input list is empty!" << std::endl;
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        for (int i = 0; i < input_list.size(); i++)
        {
            std::string tensor_name = input_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                std::cout << "tensor " << tensor_name <<" not exists!" << std::endl;
                return false;
            }
            input_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindInputs(input_tensors);
        return true;
    }

    bool TimVXEngine::bind_outputs(const std::string &op_name, const std::vector<std::string> &output_list)
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            std::cout << "op " << op_name <<" not exists!" << std::endl;
            return false;
        }
        if (output_list.size() <= 0)
        {
            std::cout << "bind output list is empty!" << std::endl;
            return false;
        }
        std::vector<std::shared_ptr<Tensor>> output_tensors;
        for (int i = 0; i < output_list.size(); i++)
        {
            std::string tensor_name = output_list[i];
            if (m_tensors.find(tensor_name) == m_tensors.end())
            {
                std::cout << "tensor " << tensor_name <<" not exists!" << std::endl;
                return false;
            }
            output_tensors.push_back(m_tensors[tensor_name]);
        }
        Operation* op_node = m_operations[op_name];
        op_node->BindOutputs(output_tensors);
        return true;
    }

    bool TimVXEngine::bind_input(const std::string &op_name, const std::string &input_name)
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            std::cout << "op " << op_name <<" not exists!" << std::endl;
            return false;
        }
        if (m_tensors.find(input_name) == m_tensors.end())
        {
            std::cout << "tensor " << input_name <<" not exists!" << std::endl;
            return false;
        }
        std::shared_ptr<Tensor> input_tensor = m_tensors[input_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindInput(input_tensor);
        return true;
    }
    bool TimVXEngine::bind_output(const std::string &op_name, const std::string &output_name)
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        if (m_operations.find(op_name) == m_operations.end())
        {
            std::cout << "op " << op_name <<" not exists!" << std::endl;
            return false;
        }
        if (m_tensors.find(output_name) == m_tensors.end())
        {
            std::cout << "tensor " << output_name <<" not exists!" << std::endl;
            return false;
        }
        std::shared_ptr<Tensor> out_tensor = m_tensors[output_name];
        Operation* op_node = m_operations[op_name];
        op_node->BindOutput(out_tensor);
        return true;
    }
    
    
    // bool TimVXEngine::set_rounding_policy(const std::string &op_name, const py::dict &rounding_policy)
    // {
    //     if (m_graph.get() == nullptr)
    //     {
    //         std::cout << "graph is invalid, please create graph first!" << std::endl;
    //         return false;
    //     }
    //     if (m_operations.find(op_name) == m_operations.end())
    //     {
    //         std::cout << "op " << op_name <<" not exists!" << std::endl;
    //         return false;
    //     }
    //     return true;
    // }

    py::dict TimVXEngine::get_op_info(const std::string &op_name)
    {
        if (m_op_info.find(op_name) != m_op_info.end())
        {
            std::cout << "op " << op_name <<" not exists!" << std::endl;
            py::dict ret;
            return ret;
        }
        return m_op_info[op_name];
    }

    bool TimVXEngine::create_graph()
    {
        m_context = tim::vx::Context::Create();
        if (nullptr == m_context.get())
        {
            std::cout << "create context fail!" << std::endl;
            return false;
        }
        m_graph = m_context->CreateGraph();
        if (nullptr == m_graph.get())
        {
            std::cout << "create graph fail!" << std::endl;
            m_context.reset();
            return false;
        }
        return true;
    }

    bool TimVXEngine::compile_graph()
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        return m_graph->Compile();
    }

    bool TimVXEngine::run_graph()
    {
        if (m_graph.get() == nullptr)
        {
            std::cout << "graph is invalid, please create graph first!" << std::endl;
            return false;
        }
        return m_graph->Run();
    }

    std::string TimVXEngine::get_graph_name()
    {
        return m_graph_name;
    }
} //namespace TIMVXPY
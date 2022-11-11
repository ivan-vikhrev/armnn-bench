#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <iostream>
#include <string>
#include <vector>

template <typename Enumeration>
auto log_as_int(Enumeration value) -> typename std::underlying_type<Enumeration>::type {
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

// armnn::Tensor create_random_tensor() {}

// std::vector<float> get_random_data() {}

int main(int argv, char **argc) {
    // Import the TensorFlow lite model.
    std::string model_path = argc[1];
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    std::cout << "Creating network from file " << model_path << std::endl;
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(model_path.c_str());

    std::cout << "Network inputs: " << std::endl;
    std::vector<armnnTfLiteParser::BindingPointInfo> input_binding_info;
    std::vector<std::string> input_names = parser->GetSubgraphInputTensorNames(0);
    for (auto input_name : input_names) {
        input_binding_info.push_back(std::move(parser->GetNetworkInputBindingInfo(0, input_name)));
        std::cout << "\t" << input_name << std::endl;
    }

    std::cout << "Network outputs: " << std::endl;
    std::vector<armnnTfLiteParser::BindingPointInfo> output_binding_info;
    std::vector<std::string> output_names = parser->GetSubgraphOutputTensorNames(0);
    for (auto output_name : output_names) {
        output_binding_info.push_back(std::move(parser->GetNetworkOutputBindingInfo(0, output_name)));
        std::cout << "\t" << output_name << std::endl;
    }

    // optimize the network.
    std::vector<std::string> error_msgs;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());
    std::cout << "Optimizing network" << std::endl;
    armnn::IOptimizedNetworkPtr opt_network = Optimize(*network,
                                                       {"CpuRef", "CpuAcc"},
                                                       runtime->GetDeviceSpec(),
                                                       armnn::OptimizerOptions(),
                                                       armnn::Optional<std::vector<std::string> &>(error_msgs));

    if (!opt_network) {
        const std::string err{"Failed to optimize network"};
        ARMNN_LOG(error) << err;
        throw armnn::Exception(err);
    }

    std::string err_message;
    armnn::NetworkId net_id{};
    std::cout << "Loading network to the CPU device" << std::endl;
    if (armnn::Status::Success != runtime->LoadNetwork(net_id, std::move(opt_network), err_message)) {
        ARMNN_LOG(error) << err_message;
        throw armnn::Exception(err_message);
    }

    std::cout << "Pre-allocating memory for output" << std::endl;
    armnn::InputTensors input_tensors;
    armnn::OutputTensors output_tensors;
    std::vector<std::vector<float>> output_buffer;
    // pre-allocate memory for output (the size of it never changes)
    for (int i = 0; i < (int)output_names.size(); ++i) {
        // const armnn::DataType data_type = output_binding_info[i].second.GetDataType();
        const armnn::TensorShape &tensor_shape = output_binding_info[i].second.GetShape();

        std::vector<float> layout_res;
        layout_res.resize(tensor_shape.GetNumElements(), 0);
        output_buffer.emplace_back(layout_res);

        // Make ArmNN output tensors
        output_tensors.reserve(output_buffer.size());
        for (size_t j = 0; j < output_buffer.size(); ++j) {
            output_tensors.emplace_back(
                std::make_pair(output_binding_info[j].first,
                               armnn::Tensor(output_binding_info[j].second, output_buffer.at(j).data())));
        }
    }

    std::cout << "Preparing tensors" << std::endl;
    const armnn::TensorShape &input_tensor_shape = input_binding_info[0].second.GetShape();
    for (int i = 0; i < (int)input_tensor_shape.GetNumDimensions(); ++i) {
        std::cout << input_tensor_shape[i] << " ";
    }
    std::cout << std::endl;
    input_tensors.clear();
    input_tensors = {{input_binding_info[0].first, armnn::ConstTensor(input_binding_info[0].second, nullptr)}};
    std::cout << "Running network" << std::endl;
    armnn::Status ret = runtime->EnqueueWorkload(net_id, input_tensors, output_tensors);
    std::cout << "Inference finished with code {" << log_as_int(ret) << "}" << std::endl;
    if (ret == armnn::Status::Failure) {
        std::string err_message = "Failed to perform inference.";
        ARMNN_LOG(error) << err_message;
        throw armnn::Exception(err_message);
    }
    return 0;
}

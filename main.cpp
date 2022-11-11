#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <iostream>
#include <vector>

int main(int argv, char **argc) {
    // Import the TensorFlow lite model.
    std::string model_path = argc[1];
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    std::cout << "Create network from file " << model_path << std::endl;
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(model_path.c_str());

    std::vector<std::string> input_names = parser->GetSubgraphInputTensorNames(0);
    std::vector<std::string> output_names = parser->GetSubgraphOutputTensorNames(0);
    std::cout << "Network inputs: " << std::endl;
    for (auto input_name : input_names) {
        std::cout << "\t" << input_name << std::endl;
    }
    std::cout << "Network outputs: " << std::endl;
    for (auto output_name : output_names) {
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

    return 0;
}

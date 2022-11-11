#include "args_handler.hpp"
#include "logger.hpp"
#include "statistics.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using HighresClock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

namespace {

std::map<armnn::DataType, std::string> str_types = {{armnn::DataType::BFloat16, "BFloat16"},
                                                    {armnn::DataType::QAsymmU8, "QAsymmU8"},
                                                    {armnn::DataType::QSymmS8, "QSymmS8QAsymmU8"},
                                                    {armnn::DataType::QSymmS16, "QSymmS16"},
                                                    {armnn::DataType::QAsymmS8, "QAsymmS8"},
                                                    {armnn::DataType::Float16, "FP16"},
                                                    {armnn::DataType::Float32, "FP32"},
                                                    {armnn::DataType::Signed32, "S32"},
                                                    {armnn::DataType::Signed64, "S64"},
                                                    {armnn::DataType::Boolean, "BOOL"}};

template <typename Enumeration>
auto log_as_int(Enumeration value) -> typename std::underlying_type<Enumeration>::type {
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}
template <typename T>
using UniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <class T>
std::vector<T> get_random_data(size_t tensor_size,
                               T rand_min = std::numeric_limits<uint8_t>::min(),
                               T rand_max = std::numeric_limits<uint8_t>::max()) {
    logger::info << "Randomly generated data" << logger::endl;
    std::vector<T> tensor_data(tensor_size);
    std::mt19937 gen(0);
    UniformDistribution<T> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; ++i) {
        tensor_data[i] = static_cast<T>(distribution(gen));
    }
    return tensor_data;
}

inline uint64_t sec_to_ns(uint32_t duration) {
    return duration * 1000000000LL;
}

inline double ns_to_ms(std::chrono::nanoseconds duration) {
    return static_cast<double>(duration.count()) * 0.000001;
}

std::string format_double(const double number) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << number;
    return ss.str();
}

} // namespace

int main(int argv, char **argc) {
    try {
        // Import the TensorFlow lite model.
        std::string model_path = argc[1];
        armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
        logger::info << "Creating network from file " << model_path << logger::endl;
        armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(model_path.c_str());

        logger::info << "Network inputs: " << logger::endl;
        std::vector<armnnTfLiteParser::BindingPointInfo> input_binding_info;
        std::vector<std::string> input_names = parser->GetSubgraphInputTensorNames(0);
        std::vector<armnn::TensorShape> input_shapes;
        for (auto input_name : input_names) {
            input_binding_info.push_back(std::move(parser->GetNetworkInputBindingInfo(0, input_name)));
            const armnn::DataType data_type = input_binding_info.back().second.GetDataType();
            const armnn::TensorShape input_shape = input_binding_info.back().second.GetShape();
            input_shapes.push_back(input_shape);

            logger::info << "\t" << input_name << " " << str_types.at(data_type) << " [";
            int i = 0;
            for (; i < (int)input_shape.GetNumDimensions() - 1; ++i) {
                logger::info << input_shape[i] << ",";
            }
            logger::info << input_shape[i] << "]" << logger::endl;
        }

        logger::info << "Network outputs: " << logger::endl;
        std::vector<armnnTfLiteParser::BindingPointInfo> output_binding_info;
        std::vector<std::string> output_names = parser->GetSubgraphOutputTensorNames(0);
        std::vector<armnn::TensorShape> output_shapes;
        for (auto output_name : output_names) {
            output_binding_info.push_back(std::move(parser->GetNetworkOutputBindingInfo(0, output_name)));
            const armnn::DataType data_type = output_binding_info.back().second.GetDataType();
            const armnn::TensorShape output_shape = output_binding_info.back().second.GetShape();
            output_shapes.push_back(output_shape);

            logger::info << "\t" << output_name << " " << str_types.at(data_type) << " [";
            int i = 0;
            for (; i < (int)output_shape.GetNumDimensions() - 1; ++i) {
                logger::info << output_shape[i] << ",";
            }
            logger::info << output_shape[i] << "]" << logger::endl;
        }

        // optimize the network.
        std::vector<std::string> error_msgs;
        armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());
        logger::info << "Optimizing network" << logger::endl;
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
        logger::info << "Loading network to the CPU device" << logger::endl;
        if (armnn::Status::Success != runtime->LoadNetwork(net_id, std::move(opt_network), err_message)) {
            ARMNN_LOG(error) << err_message;
            throw armnn::Exception(err_message);
        }

        logger::info << "Pre-allocating memory for output" << logger::endl;
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

        logger::info << "Preparing tensors" << logger::endl;
        int ntensors = 10;
        std::vector<armnn::InputTensors> inputs;
        std::vector<std::vector<float>> input_beffers;
        const armnn::TensorShape &input_tensor_shape = input_binding_info[0].second.GetShape();
        for (int i = 0; i < ntensors; ++i) {
            input_beffers.push_back(get_random_data<float>(input_tensor_shape.GetNumElements(), -1, 1));
            inputs.push_back({{input_binding_info[0].first,
                               armnn::ConstTensor(input_binding_info[0].second, input_beffers.back().data())}});
        }
        // set time limit
        uint32_t time_limit_sec = 60;
        if (time_limit_sec == 0) {
            time_limit_sec = 60;
        }
        uint64_t time_limit_ns = sec_to_ns(time_limit_sec);

        logger::info << "Measuring model performance" << logger::endl;

        auto infer_start_time = HighresClock::now();
        armnn::Status ret = runtime->EnqueueWorkload(net_id, inputs[0], output_tensors);
        auto first_inference_time = ns_to_ms(HighresClock::now() - infer_start_time);
        if (ret == armnn::Status::Failure) {
            std::string err_message = "Failed to perform inference.";
            ARMNN_LOG(error) << err_message;
            throw armnn::Exception(err_message);
        }
        logger::info << "Warming up inference took " << format_double(first_inference_time) << " ms" << logger::endl;

        std::vector<double> latencies;
        int64_t iteration = 0;
        auto start_time = HighresClock::now();
        auto uptime = std::chrono::duration_cast<ns>(HighresClock::now() - start_time).count();
        while (static_cast<uint64_t>(uptime) < time_limit_ns) {
            infer_start_time = HighresClock::now();
            runtime->EnqueueWorkload(net_id, inputs[iteration % inputs.size()], output_tensors);
            latencies.push_back(ns_to_ms(HighresClock::now() - infer_start_time));
            uptime = std::chrono::duration_cast<ns>(HighresClock::now() - start_time).count();
            ++iteration;
        }

        // check outputs
        // auto output = module.forward(inputs[0]).toTuple()->elements()[0].toTensor();
        // logger::info << output << logger::endl;
        Metrics metrics(latencies, 1);

        // Performance metrics report
        logger::info << "Count: " << iteration << " iterations" << logger::endl;
        logger::info << "Duration: " << format_double(uptime * 0.000001) << " ms" << logger::endl;
        logger::info << "Latency:" << logger::endl;
        logger::info << "\tMedian   " << format_double(metrics.latency.median) << " ms" << logger::endl;
        logger::info << "\tAverage: " << format_double(metrics.latency.avg) << " ms" << logger::endl;
        logger::info << "\tMin:     " << format_double(metrics.latency.min) << " ms" << logger::endl;
        logger::info << "\tMax:     " << format_double(metrics.latency.max) << " ms" << logger::endl;
        logger::info << "Throughput: " << format_double(metrics.fps) << " FPS" << logger::endl;
    } catch (const std::exception &ex) {
        logger::err << ex.what() << logger::endl;
        return -1;
    }
    return 0;
}

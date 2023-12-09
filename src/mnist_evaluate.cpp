#include <fstream>
#include <iostream>

#include "cxxopts.hpp"
#include "mnist.hpp"
#include "optimize.hpp"

int main(int arg_count, char** args) {
    cxxopts::Options options("MNISTNetEvaluator", "MNIST Character Recognition Network Evaluation Tool");

    options.add_options()("l,labels", "The test labels file.", cxxopts::value<std::string>())(
        "i,images", "The test images file.", cxxopts::value<std::string>())(
        "m,mini", "Use the mini network (for debugging)", cxxopts::value<bool>()->default_value("false"))(
        "h,help", "Print this help message")("load", "File to load starting parameters from",
                                             cxxopts::value<std::string>())(
        "c,count", "How many test samples to test before calculating stats and quitting.",
        cxxopts::value<size_t>()->default_value("0"));

    auto result = options.parse(arg_count, args);

    if (result.count("help") > 0) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    std::cout << "Loading MNIST test data...";
    Ml::MNISTLoader mnist_data(result["labels"].as<std::string>(), result["images"].as<std::string>());
    std::cout << "loaded " << mnist_data.Count() << " test images." << std::endl;

    const double mnist_std_dev = mnist_data.GetStd();
    const double mnist_mean = mnist_data.GetMean();

    std::cout << "  MNIST std deviation : " << mnist_std_dev << std::endl;
    std::cout << "  MNIST mean          : " << mnist_mean << std::endl;

    std::cout << "Building MNIST Network..." << std::endl;
    auto mnist_network = result["mini"].as<bool>() ? Ml::MNIST::MiniNetwork(mnist_data.GetPixelsPerImage())
                                                   : Ml::MNIST::Network(mnist_data.GetPixelsPerImage());

    if (result.count("load")) {
        const std::string in_file = result["load"].as<std::string>();
        std::cout << "Loading parameters from: " << in_file << std::endl;
        mnist_network->DeserializeFrom(in_file);
    } else {
        std::cout << "WARNING: Using random weights.  Did you mean to specify a --load?" << std::endl;
    }

    const size_t sample_count = mnist_data.Count();
    const size_t test_count = result["count"].as<size_t>();

    Ml::ParamVector current_params(mnist_network->GetOptimizedParamsCount());
    current_params << mnist_network->GetOptimizedParams();

    Ml::GradVector batch_gradient(current_params.rows());
    batch_gradient.fill(0);
    auto objective_fn = Ml::LogLossObjective;
    std::vector<bool> correct_classificiations;

    for (size_t training_idx = 0; training_idx < sample_count; training_idx++) {
        if (test_count > 0 && training_idx >= test_count) {
            std::cout << "Finished testing " << test_count << " samples, stopping." << std::endl;
            break;
        }

        // std::cout << "Loading training sample #" << training_idx << std::endl;
        auto [this_label, this_image] = mnist_data.GetSample(training_idx);

        // std::cout << "  Getting one hot vector..." << std::endl;
        auto this_one_hot = Ml::MNIST::GetOneHotVector(this_label);
        // std::cout << "  Getting image in input format..." << std::endl;
        auto this_image_input_format = Ml::MNIST::ImageToNormalizedInput(this_image, mnist_mean, mnist_std_dev);

        // TODO(imo): Make gradient return apply too so we don't need to double do this...
        auto this_full_node_output = mnist_network->Apply(this_image_input_format);
        auto prediction_vector = mnist_network->OutputVectorFromNodeOutputs(this_full_node_output);
        auto max_element_iterator = std::max_element(prediction_vector.begin(), prediction_vector.end());
        size_t max_element_idx = std::distance(prediction_vector.begin(), max_element_iterator);

        correct_classificiations.push_back(max_element_idx == this_label);
        std::cout << "Sample " << training_idx << " guess==actual  " << max_element_idx << "==" << this_label << " "
                  << correct_classificiations[training_idx] << std::endl;
    }

    double correct_count = 0.0;
    for (size_t idx = 0; idx < correct_classificiations.size(); idx++) {
        if (correct_classificiations[idx]) {
            correct_count += 1.0;
        }
    }

    std::cout << std::endl
              << std::endl
              << "After " << correct_classificiations.size() << " tests, " << correct_count
              << " were correct.  Success % = "
              << 100.0 * correct_count / static_cast<double>(correct_classificiations.size()) << std::endl;

    return 0;
}
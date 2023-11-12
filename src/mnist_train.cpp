#include <iostream>

#include "cxxopts.hpp"

#include "mnist.hpp"
#include "optimize.hpp"

int main(int arg_count, char** args) {
    cxxopts::Options options("MNISTNetTrainer", "MNIST Character Recognition Network Training Tool");

    options.add_options()
        ("l,labels", "The training labels file.", cxxopts::value<std::string>())
        ("i,images", "The training images file.", cxxopts::value<std::string>())
        ("s,step_size", "The training step size (gradient multiplier).", cxxopts::value<double>()->default_value("0.001"))
        ("b,beta_1", "Beta 1 in the Adam Optimizer", cxxopts::value<double>()->default_value("0.9"))
        ("B,beta_2", "Beta 2 in the Adam Optimizer", cxxopts::value<double>()->default_value("0.999"))
        ("e,epsilon", "Epsilon in the adam optimizer.", cxxopts::value<double>()->default_value("1e-8"))
        ("c,batch_size", "Number of training samples per gradient step", cxxopts::value<size_t>()->default_value("10"))
        ("h,help", "Print this help message");

    auto result = options.parse(arg_count, args);
    auto dbl_opt = [&result](std::string opt) -> double {
        return result[opt].as<double>();
    };

    if (result.count("help") > 0) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    Vs::MNISTLoader mnist_training_data(result["labels"].as<std::string>(), result["images"].as<std::string>());

    auto mnist_network = Vs::MNISTNetwork(mnist_training_data.GetPixelsPerImage());

    Vs::AdamOptimizer optimizer(dbl_opt("step_size"), dbl_opt("beta_1"), dbl_opt("beta_2"), dbl_opt("epsilon"));

    const size_t batch_size = result["batch_size"].as<size_t>();

    return 0;
}
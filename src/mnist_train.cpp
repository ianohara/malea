#include "mnist.hpp"

#include "cxxopts.hpp"

#include <iostream>

int main(int arg_count, char** args) {
    cxxopts::Options options("MNISTNetTrainer", "MNIST Character Recognition Network Training Tool");

    options.add_options()
        ("l,labels", "The training labels file.", cxxopts::value<std::string>())
        ("i,images", "The training images file.", cxxopts::value<std::string>())
        ("h,help", "Print this help message");

    auto result = options.parse(arg_count, args);

    if (result.count("help") > 0) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    Vs::MNISTLoader mnist_training_data(result["labels"].as<std::string>(), result["images"].as<std::string>());

    return 0;
}
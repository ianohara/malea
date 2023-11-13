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

    std::cout << "Loading MNIST training data..." << std::endl;
    Vs::MNISTLoader mnist_training_data(result["labels"].as<std::string>(), result["images"].as<std::string>());

    std::cout << "Building MNIST Network..." << std::endl;
    auto mnist_network = Vs::MNIST::Network(mnist_training_data.GetPixelsPerImage());

    Vs::AdamOptimizer optimizer(dbl_opt("step_size"), dbl_opt("beta_1"), dbl_opt("beta_2"), dbl_opt("epsilon"), mnist_network->GetOptimizedWeightCount());

    const size_t batch_size = result["batch_size"].as<size_t>();
    const size_t training_count = mnist_training_data.Count();

    Vs::ParamVector current_params(mnist_network->GetOptimizedWeightCount());

    size_t count_in_batch = 0;
    Vs::FVal batch_error = 0;
    Vs::GradVector batch_gradient(current_params.rows());
    batch_gradient.fill(0);

    for (size_t training_idx=0; training_idx < training_count; training_idx++) {
        std::cout << "Loading training sample #" << training_idx << std::endl;
        auto[this_label, this_image] = mnist_training_data.GetSample(training_idx);

        std::cout << "  Getting one hot vector..." << std::endl;
        auto this_one_hot = Vs::MNIST::GetOneHotVector(this_label);
        std::cout << "  Getting image in input format..." << std::endl;
        auto this_image_input_format = Vs::MNIST::ImageToInput(this_image);
        std::cout << "  Calculating gradient..." << std::endl;
        batch_gradient += mnist_network->WeightGradient(this_image_input_format, this_one_hot, Vs::LogLossObjective);
        // TODO(imo): Make gradient return apply too so we don't need to double do this...
        batch_error += Vs::LogLossObjective->Apply(mnist_network->OutputVectorFromNodeOutputs(mnist_network->Apply(this_image_input_format)), this_one_hot);
        count_in_batch++;

        if (count_in_batch >= batch_size || training_idx >= training_count) {
            std::cout << "  Batch ready, taking step (Batch error=" << batch_error << ")..." << std::endl;
            std::cout << "    Averaging gradient..." << std::endl;
            batch_gradient /= static_cast<Vs::FVal>(count_in_batch);

            std::cout << "    Taking step..." << std::endl;
            current_params = optimizer.Step(current_params, batch_gradient);
            std::cout << "    Putting new weights into network..." << std::endl;
            mnist_network->SetOptimizedWeights(current_params);
            std::cout << "Completed step " << optimizer.GetStepCount() << std::endl;

            batch_error = 0;
            batch_gradient.fill(0);
            count_in_batch = 0;
        }
    }

    return 0;
}
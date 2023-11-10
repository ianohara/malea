#include <memory>
#include <iostream>

#include "cxxopts.hpp"

#include "network.hpp"
#include "optimize.hpp"
#include "util.hpp"

std::shared_ptr<Vs::Network> XORNetwork() {
    auto net = std::make_shared<Vs::Network>(2);
    net->AddFullyConnectedLayer(2, Vs::ReLu);
    net->AddFullyConnectedLayer(2, Vs::ReLu);
    net->AddFullyConnectedLayer(1, Vs::ReLu);

    net->SetWeightsWith([](size_t from, size_t to) -> double { return Vs::Util::RandInRange(-5.0, 5.0); });

    return net;
}

int main(int arg_count, char** args) {
    cxxopts::Options options("XOR Trainer", "XOR Network Training Tool");

    options.add_options()
        ("s,step_size", "The training step size (gradient multiplier).", cxxopts::value<double>()->default_value("0.001"))
        ("b,beta_1", "Beta 1 in the Adam Optimizer", cxxopts::value<double>()->default_value("0.9"))
        ("B,beta_2", "Beta 2 in the Adam Optimizer", cxxopts::value<double>()->default_value("0.999"))
        ("e,epsilon", "Epsilon in the adam optimizer.", cxxopts::value<double>()->default_value("1e-8"))
        ("c,epoch_count", "Epochs to train for.", cxxopts::value<size_t>()->default_value("1000"))
        ("h,help", "Print this help message");

    auto result = options.parse(arg_count, args);
    auto dbl_opt = [&result](std::string opt) -> double {
        return result[opt].as<double>();
    };

    if (result.count("help") > 0) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }


    std::cout << "Building XOR Network..." << std::endl;
    auto xor_network = XORNetwork();

    Vs::AdamOptimizer optimizer(dbl_opt("step_size"), dbl_opt("beta_1"), dbl_opt("beta_2"), dbl_opt("epsilon"), xor_network->GetOptimizedParamsCount());

    Vs::ParamVector current_params(xor_network->GetOptimizedParamsCount());
    current_params << xor_network->GetOptimizedParams();

    size_t epochs_to_train = result["epoch_count"].as<size_t>();
    // Setup epoch data so that each column contains the inputs in rows 0 and 1, and the expected result in row 3.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> epoch_data(3, 4);
    epoch_data <<
        1, 0, 1, 0,
        1, 1, 0, 0,
        0, 1, 1, 0;

    epoch_data = epoch_data.reshaped(3, 4);

    const size_t samples_per_epoch =  epoch_data.cols();
    Vs::GradVector epoch_gradient(current_params.rows());
    double epoch_error;

    for (size_t epoch = 0; epoch < epochs_to_train; epoch++) {
        std::cout << "Starting epoch " << epoch << " with " << samples_per_epoch << " samples" << std::endl;
        if (epoch != 0 && epoch_error <= 0.00001) {
            std::cout << "Last epoch error was small, done." << std::endl;
            return 0;
        }

        if (epoch != 0 && epoch_gradient.norm() < 0.00001) {
            std::cout << "Gradient has diminished to nothing, stopping." << std::endl;
            return 0;
        }

        epoch_error = 0;
        epoch_gradient.fill(0);
        for (size_t sample_idx = 0; sample_idx < samples_per_epoch; sample_idx++) {
            Vs::IOVector input = epoch_data(Eigen::seq(0, 1), sample_idx).array() - 0.5;
            Vs::IOVector expected_output = epoch_data(Eigen::seq(2, 2), sample_idx).array() - 0.5;

            Vs::GradVector sample_gradient = xor_network->WeightGradient(input, expected_output, Vs::SumOfSquaresObjective);
            // std::cout << "  sample_gradient norm is " << sample_gradient.norm() << std::endl;
            epoch_gradient += sample_gradient;
            // TODO(imo): Make gradient return apply too so we don't need to double do this...
            auto this_full_node_output = xor_network->Apply(input);
            auto prediction_vector = xor_network->OutputVectorFromNodeOutputs(this_full_node_output);
            double sample_error = Vs::SumOfSquaresObjective->Apply(prediction_vector, expected_output);
            epoch_error += sample_error;
            std::cout << "  Sample " << sample_idx << ":" << std::endl
                << "    Input        : " << input.transpose() << std::endl
                << "    Expected Out : " << expected_output << std::endl
                << "    Predicted Out: " << prediction_vector << std::endl
                << "    Sample Error : " << sample_error << std::endl;
        }

        std::cout << "  Done epoch " << epoch << " with error=" << epoch_error << "..." << std::endl;
        epoch_gradient /= static_cast<Vs::FVal>(samples_per_epoch);
        // xor_network->SummarizeNonZeroParams(std::cout);
        // xor_network->SummarizeParamGradient(std::cout, epoch_gradient);
        // xor_network->SummarizeObjDelNode(std::cout);
        // // xor_network->SummarizeNodeOutputs(std::cout, this_image_input_format, false);
        // xor_network->SummarizeWeightsForLayer(std::cout, 1, 10);
        // std::cout << "    Taking step..." << std::endl;
        current_params = optimizer.Step(current_params, epoch_gradient);
        // std::cout << "    Putting new weights into network..." << std::endl;
        xor_network->SetOptimizedParams(current_params);
    }

    return 0;
}
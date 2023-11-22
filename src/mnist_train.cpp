#include <iostream>
#include <fstream>

#include "cxxopts.hpp"
#include "mnist.hpp"
#include "optimize.hpp"

int main(int arg_count, char** args) {
    cxxopts::Options options("MNISTNetTrainer", "MNIST Character Recognition Network Training Tool");

    options.add_options()("l,labels", "The training labels file.", cxxopts::value<std::string>())(
        "i,images", "The training images file.", cxxopts::value<std::string>())(
        "s,step_size", "The training step size (gradient multiplier).",
        cxxopts::value<double>()->default_value("0.001"))("b,beta_1", "Beta 1 in the Adam Optimizer",
                                                          cxxopts::value<double>()->default_value("0.9"))(
        "B,beta_2", "Beta 2 in the Adam Optimizer", cxxopts::value<double>()->default_value("0.999"))(
        "e,epsilon", "Epsilon in the adam optimizer.", cxxopts::value<double>()->default_value("1e-8"))(
        "c,batch_size", "Number of training samples per gradient step", cxxopts::value<size_t>()->default_value("10"))(
        "m,mini", "Use the mini network (for debugging)", cxxopts::value<bool>()->default_value("false"))(
        "h,help", "Print this help message")(
        "l,load", "File to load starting parameters from", cxxopts::value<std::string>())(
        "o,out", "File to write params to after each batch", cxxopts::value<std::string>()->default_value("/tmp/mnist_params.bin"));

    auto result = options.parse(arg_count, args);
    auto dbl_opt = [&result](std::string opt) -> double { return result[opt].as<double>(); };

    if (result.count("help") > 0) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    std::cout << "Loading MNIST training data...";
    Vs::MNISTLoader mnist_training_data(result["labels"].as<std::string>(), result["images"].as<std::string>());
    std::cout << "loaded " << mnist_training_data.Count() << " training images." << std::endl;

    const double mnist_std_dev = mnist_training_data.GetStd();
    const double mnist_mean = mnist_training_data.GetMean();

    std::cout << "  MNIST std deviation : " << mnist_std_dev << std::endl;
    std::cout << "  MNIST mean          : " << mnist_mean << std::endl;

    std::cout << "Building MNIST Network..." << std::endl;
    auto mnist_network = result["mini"].as<bool>() ? Vs::MNIST::MiniNetwork(mnist_training_data.GetPixelsPerImage())
                                                   : Vs::MNIST::Network(mnist_training_data.GetPixelsPerImage());

    if (result.count("load")) {
        const std::string in_file = result["load"].as<std::string>();
        std::cout << "Loading parameters from: " << in_file << std::endl;
        mnist_network->DeserializeFrom(in_file);
    }

    Vs::AdamOptimizer optimizer(dbl_opt("step_size"), dbl_opt("beta_1"), dbl_opt("beta_2"), dbl_opt("epsilon"),
                                mnist_network->GetOptimizedParamsCount());

    const size_t batch_size = result["batch_size"].as<size_t>();
    const size_t training_count = mnist_training_data.Count();

    Vs::ParamVector current_params(mnist_network->GetOptimizedParamsCount());
    current_params << mnist_network->GetOptimizedParams();

    size_t count_in_batch = 0;
    Vs::FVal batch_error = 0;
    Vs::GradVector batch_gradient(current_params.rows());
    batch_gradient.fill(0);
    auto objective_fn = Vs::LogLossObjective;
    for (size_t training_idx = 0; training_idx < training_count; training_idx++) {
        // std::cout << "Loading training sample #" << training_idx << std::endl;
        auto [this_label, this_image] = mnist_training_data.GetSample(training_idx);

        // std::cout << "  Getting one hot vector..." << std::endl;
        auto this_one_hot = Vs::MNIST::GetOneHotVector(this_label);
        // std::cout << "  Getting image in input format..." << std::endl;
        auto this_image_input_format = Vs::MNIST::ImageToNormalizedInput(this_image, mnist_mean, mnist_std_dev);
        // std::cout << "  Calculating gradient..." << std::endl;
        Vs::GradVector sample_gradient =
            mnist_network->WeightGradient(this_image_input_format, this_one_hot, objective_fn);
        // std::cout << "  sample_gradient norm is " << sample_gradient.norm() << std::endl;
        batch_gradient += sample_gradient;
        // TODO(imo): Make gradient return apply too so we don't need to double do this...
        auto this_full_node_output = mnist_network->Apply(this_image_input_format);
        auto prediction_vector = mnist_network->OutputVectorFromNodeOutputs(this_full_node_output);
        batch_error += objective_fn->Apply(prediction_vector, this_one_hot);
        count_in_batch++;

        if (count_in_batch >= batch_size || training_idx >= training_count) {
            std::cout << "  Batch ready, taking step (Batch error=" << batch_error << ")..." << std::endl;
            std::cout << "    Last one hot (label=" << this_label << ") vs prediction: " << std::endl
                      << "      " << this_one_hot.transpose() << std::endl
                      << "      " << prediction_vector.transpose() << std::endl;

            // std::cout << "    Averaging gradient..." << std::endl;
            batch_gradient /= static_cast<Vs::FVal>(count_in_batch);
            // std::cout << "    batch gradient norm is=" << batch_gradient.norm() << std::endl;
            mnist_network->SummarizeNonZeroParams(std::cout);
            // mnist_network->SummarizeParamGradient(std::cout, batch_gradient);
            // mnist_network->SummarizeObjDelNode(std::cout);
            // mnist_network->SummarizeNodeOutputs(std::cout, this_image_input_format, false);
            //  mnist_network->SummarizeWeightsForLayer(std::cout, 1, 10);
            //  std::cout << "    Taking step..." << std::endl;
            current_params = optimizer.Step(current_params, batch_gradient);
            // std::cout << "    Putting new weights into network..." << std::endl;
            mnist_network->SetOptimizedParams(current_params);

            if (result.count("out")) {
                const std::string out_file = result["out"].as<std::string>();
                std::cout << "Writing parameters to: " << out_file << std::endl;
                mnist_network->SerializeTo(out_file);
            }

            std::cout << "Completed step " << optimizer.GetStepCount()
                      << " batch error / # samples in batch=" << batch_error / count_in_batch << std::endl;

            batch_error = 0;
            batch_gradient.fill(0);
            count_in_batch = 0;
        }
    }

    return 0;
}
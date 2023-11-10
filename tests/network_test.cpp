#include "gtest/gtest.h"

#include "network.hpp"

#include <cstdlib>
#include <tuple>

static constexpr Vs::FVal NetEps = 0.0001;

namespace VsTest {
    static double RandInRange(double min, double max) {
        return min + (max - min) * (std::rand() / static_cast<double>(RAND_MAX));
    }
}

TEST(Network, Create)
{
    Vs::Network n(200);
}

TEST(Network, FullyConnected)
{
    Vs::Network n(5);
    n.AddFullyConnectedLayer(3, Vs::ReLu);
    n.AddFullyConnectedLayer(4, Vs::ReLu);
}

TEST(Network, SingleNodePassThroughLayers)
{
    Vs::Network n(1);
    n.AddFullyConnectedLayer(1, Vs::PassThrough);
    n.AddFullyConnectedLayer(1, Vs::PassThrough);
    n.AddFullyConnectedLayer(1, Vs::PassThrough);
    n.SetUnityWeights();

    Vs::IOVector input(1);
    input << 101.1;

    auto node_outputs = n.Apply(input);
    ASSERT_NEAR(n.OutputVectorFromNodeOutputs(node_outputs).sum(), 101.1, NetEps);
}

TEST(Network, SingleNodeReLuLayers)
{
    Vs::Network n(1);
    n.AddFullyConnectedLayer(1, Vs::ReLu);
    n.AddFullyConnectedLayer(1, Vs::ReLu);
    n.AddFullyConnectedLayer(1, Vs::ReLu);
    n.SetUnityWeights();

    Vs::IOVector input(1);
    input << -1.0;

    auto node_outputs = n.Apply(input);
    ASSERT_NEAR(n.OutputVectorFromNodeOutputs(node_outputs).sum(), 0, NetEps);

    input << 2.0;
    node_outputs = n.Apply(input);
    ASSERT_NEAR(n.OutputVectorFromNodeOutputs(node_outputs).sum(), 2.0, NetEps);
}

TEST(Network, TwoNodePassThroughLayers)
{
    Vs::Network n(2);
    n.AddFullyConnectedLayer(2, Vs::PassThrough);
    n.AddFullyConnectedLayer(2, Vs::PassThrough);
    n.AddFullyConnectedLayer(2, Vs::PassThrough);
    n.AddFullyConnectedLayer(2, Vs::PassThrough);
    n.SetUnityWeights();

    Vs::IOVector input(2);

    input << 1, 1;
    auto node_outputs = n.Apply(input);
    ASSERT_NEAR(n.OutputVectorFromNodeOutputs(node_outputs).sum(), 32, NetEps);
}

TEST(Network, BigHonkerReLu)
{
    const size_t honkin_size = 250;
    Vs::Network n(honkin_size);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.SetAllWeightsTo(1.0 / honkin_size); // So each layer adds up to the previous output

    Vs::IOVector input(honkin_size);
    input.fill(123.0);

    auto node_outputs = n.Apply(input);
    ASSERT_NEAR(n.OutputVectorFromNodeOutputs(node_outputs).sum(), honkin_size * 123.0, NetEps);
}

TEST(Networm, SingleNodeSoftMax) {
    Vs::Network n(1);
    n.AddFullyConnectedLayer(1, Vs::ReLu);
    n.AddSoftMaxLayer();
    n.SetUnityWeights();

    Vs::IOVector input(1);

    // Regardless of input this should always come out of a single node softmax as 1.
    Vs::IOVector expected(1);
    expected << 1.0;
    for (size_t i = 0; i < 10; i++) {
        input << VsTest::RandInRange(-1e6, 1e6);
        auto all_output = n.Apply(input);
        auto output = n.OutputVectorFromNodeOutputs(all_output);
        ASSERT_NEAR(output(0), expected(0), NetEps) << "input=" << input.transpose() << " output=" << output.transpose() << " expected=" << expected << std::endl;
    }

}

TEST(Network, FiveNodeSoftMax)
{
    Vs::Network n(5);
    n.AddSoftMaxLayer();

    Vs::IOVector input(5), expected(5);

    auto check = [&]() {
        auto all_node_outputs = n.Apply(input);
        auto output = n.OutputVectorFromNodeOutputs(all_node_outputs);

        ASSERT_EQ(output.size(), expected.size());
        for (int i=0; i < 5; i++) {
            ASSERT_NEAR(output(i), expected(i), NetEps)
                << "output=" << output.transpose() << " doesn't match expected="
                << expected.transpose() << " at least at index " << i << std::endl;
        }
    };

    input << 0, 0, 0, 1, 0;
    expected << 0.148848, 0.148848, 0.148848, 0.40461, 0.148848;
    check();

    input << 1000, 200000, 10, 20, 3.14;
    expected << 0, 1, 0, 0, 0;
    check();

    input << -0.164209, 0.61607, -0.331446, -0.103701, 0.141333;
    expected << 0.15509126, 0.33842169, 0.13120706, 0.16476525, 0.21051475;
    check();
}

TEST(Network, GradientFiveNodeSoftMax)
{
    Vs::Network n(5);
    n.AddFullyConnectedLayer(5, Vs::ReLu);
    n.AddSoftMaxLayer();
    n.SetUnityWeights();

    Vs::IOVector input(5), expected(5);
    input << 0.1, 0.2, 0.3, 0.4, 0.5;
    expected << 0, 0, 0, 0, 1;

    auto weight_gradient = n.WeightGradient(input, expected, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientSingleNodeLayers)
{
    const size_t num_layers = 4;
    Vs::Network n(1);

    for (auto layer_idx = 0; layer_idx < num_layers; layer_idx++)
    {
        n.AddFullyConnectedLayer(1, Vs::ReLu);
    }
    n.SetUnityWeights();

    Vs::IOVector input(1);
    input.fill(1);

    auto out = n.WeightGradient(input, input, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientMultipleNodeMultipleLayer)
{
    const size_t num_layers = 4;
    Vs::Network n(3);

    for (auto layer_idx = 0; layer_idx < num_layers; layer_idx++)
    {
        n.AddFullyConnectedLayer(3, Vs::ReLu);
    }
    n.SetUnityWeights();

    Vs::IOVector input(3);
    input.fill(0.5);

    // Just make sure it runs!
    auto out = n.WeightGradient(input, 2 * input, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientBigHonkin)
{
    const size_t honkin_size = 250;
    Vs::Network n(honkin_size);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);
    n.AddFullyConnectedLayer(honkin_size, Vs::ReLu);

    auto input = Vs::IOVector(honkin_size);
    input.fill(1);

    // Just make sure it runs!
    auto out = n.WeightGradient(input, 0.1 * input, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientHandCalcChecks)
{
    Vs::Network n(1);
    n.AddFullyConnectedLayer(2, Vs::ArgCubed);
    n.AddFullyConnectedLayer(2, Vs::ArgCubed);
    n.AddFullyConnectedLayer(2, Vs::ArgCubed);

    auto calc_dels = [&n](Vs::IOVector &actual_outputs, Vs::IOVector &expected_outputs)
    {
        auto i_5 = n.GetWeightForConnection(3, 5) * actual_outputs[3] + n.GetWeightForConnection(4, 5) * actual_outputs[4];
        auto i_6 = n.GetWeightForConnection(3, 6) * actual_outputs[3] + n.GetWeightForConnection(4, 6) * actual_outputs[4];
        auto i_3 = n.GetWeightForConnection(1, 3) * actual_outputs[1] + n.GetWeightForConnection(2, 3) * actual_outputs[2];
        Vs::FVal del_E_del_w_35 = -2.0 * (expected_outputs[0] - actual_outputs[5]) * 3.0 * i_5 * i_5 * actual_outputs[3];
        Vs::FVal del_E_del_w_13 = -2.0 * (expected_outputs[0] - actual_outputs[5]) * 3.0 * i_5 * i_5 * (n.GetWeightForConnection(3, 5) * 3.0 * i_3 * i_3 * actual_outputs[1])
            - 2.0 * (expected_outputs[1] - actual_outputs[6]) * (3.0 * i_6 * i_6) * (n.GetWeightForConnection(3, 6) * 3.0 * i_3 * i_3 * actual_outputs[1]);

        return std::make_pair(del_E_del_w_13, del_E_del_w_35);
    };

    auto test = [&n, &calc_dels](Vs::FVal in0, Vs::FVal out0, Vs::FVal out1)
    {
        Vs::IOVector input(1);
        input << in0;
        Vs::IOVector output(2);
        output << out0, out1;

        auto node_outputs = n.Apply(input);
        auto gradient = n.WeightGradient(input, output, Vs::SumOfSquaresObjective);

        auto [del_E_del_w_13, del_E_del_w_35] = calc_dels(node_outputs, output);

        auto connection_to_idx = n.GetConnectionToWeightParamIdx();
        ASSERT_NEAR(gradient(connection_to_idx[3][5]), del_E_del_w_35, NetEps);
        ASSERT_NEAR(gradient(connection_to_idx[1][3]), del_E_del_w_13, NetEps);
    };

    n.SetUnityWeights();
    test(1, 0, 0);
    test(0, 0, 0);
    test(0.1, 0.2, 0.2);
    test(1, 0.1, 0.2);
    test(0.9, 2, 0.7);
    test(0.3, 0.0001, 0.0001);

    auto rand_weights = [](size_t row, size_t col) {
        return (std::rand() / static_cast<Vs::WVal>(RAND_MAX)) - 1.0;
    };
    n.SetWeightsWith(rand_weights);

    test(1, 0, 0);
}

TEST(Network, MNISTFullyConnected)
{
    const size_t image_pixels = 28 * 28;
    Vs::Network n(image_pixels);
    n.AddFullyConnectedLayer(100, Vs::ReLu);
    n.AddFullyConnectedLayer(100, Vs::ReLu);
    n.AddFullyConnectedLayer(10, Vs::ReLu);
    n.AddSoftMaxLayer();

    n.SetUnityWeights();

    Vs::IOVector input(image_pixels);
    input.fill(10);
    Vs::IOVector expected_out(10);
    expected_out.fill(0);
    expected_out(1) = 1;

    n.WeightGradient(input, expected_out, Vs::LogLossObjective);
}

template<typename T> void AssertVecEqual(std::vector<T> actual, std::vector<T> expected) {
    GTEST_ASSERT_EQ(actual.size(), expected.size()) << "Vector sizes not equal";

    for (size_t idx = 0; idx < actual.size(); idx++) {
        GTEST_ASSERT_EQ(actual[idx], expected[idx]) << "At idx=" << idx << " Values do not match.";
    }
}

TEST(Network, GetIncomingNodesFor) {
    Vs::Network n(10);  // Input layer is 0-9
    n.AddFullyConnectedLayer(2, Vs::ReLu); // Nodes 10 and 11
    n.AddSoftMaxLayer(); // Nodes 12 and 13

    std::vector<size_t> node_0_to_9_incoming = {};
    std::vector<size_t> node_10_and_11_incoming = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (size_t idx = 0; idx <= 9; idx++) {
        AssertVecEqual(n.GetIncomingNodesFor(idx), node_0_to_9_incoming);
    }

    AssertVecEqual(n.GetIncomingNodesFor(10), node_10_and_11_incoming);
    AssertVecEqual(n.GetIncomingNodesFor(11), node_10_and_11_incoming);

    AssertVecEqual(n.GetIncomingNodesFor(12), std::vector<size_t>{10});
    AssertVecEqual(n.GetIncomingNodesFor(13), std::vector<size_t>{11});
}

TEST(Network, GetOutgoingNodesFor) {
    Vs::Network n(10); // input layer is 0-9
    n.AddFullyConnectedLayer(2, Vs::ReLu); // nodes 10 and 11
    n.AddSoftMaxLayer(); // nodes 12 and 13

    std::vector<size_t> node_0_to_9_outgoing = {10, 11};
    std::vector<size_t> node_12_and_13_outgoing = {};

    for (size_t idx = 0; idx <= 9; idx++) {
        AssertVecEqual(n.GetOutgoingNodesFor(idx), node_0_to_9_outgoing);
    }

    AssertVecEqual(n.GetOutgoingNodesFor(10), std::vector<size_t>{12});
    AssertVecEqual(n.GetOutgoingNodesFor(11), std::vector<size_t>{13});

    AssertVecEqual(n.GetOutgoingNodesFor(12), node_12_and_13_outgoing);
    AssertVecEqual(n.GetOutgoingNodesFor(13), node_12_and_13_outgoing);
}

TEST(Network, AnalyticAndNumericalGradientsMatchSimple) {
    auto np = std::make_shared<Vs::Network>(1);
    np->AddFullyConnectedLayer(3, Vs::Sigmoid);
    np->AddFullyConnectedLayer(4, Vs::Sigmoid);
    np->AddFullyConnectedLayer(2, Vs::Sigmoid);

    Vs::IOVector input(1); input << 1.1;
    Vs::IOVector output(2); output << 0, 0;
    auto objective_fn = Vs::SumOfSquaresObjective;
    double param_step_size = 0.00001;

    Vs::IOVector analytic_gradient = np->WeightGradient(input, output, objective_fn);
    Vs::IOVector numerical_gradient = Vs::CalculateNumericalGradient(np, input, output, objective_fn, param_step_size);

    std::cout << std::endl << "Top 10 Differences Of Analytic Vs Numerical" << std::endl;
    Vs::Util::TopTenDifferences(std::cout, analytic_gradient, numerical_gradient, [&np](size_t row, size_t col){ std::stringstream ss; np->DescribeParamIdx(ss, row); return ss.str();});

    GTEST_ASSERT_LE((analytic_gradient - numerical_gradient).norm(), 0.01);
}

TEST(Network, AnalyticAndNumericalGradientsMatchBig) {
    auto np = std::make_shared<Vs::Network>(10);
    np->AddFullyConnectedLayer(25, Vs::ReLu);
    np->AddFullyConnectedLayer(10, Vs::ReLu);

    auto norm_dist_vec = [](size_t dim) {
        Vs::IOVector vec(dim);
        for (size_t idx = 0; idx < dim; idx++) {
            vec(idx) = Vs::Util::RandInGaussian(0.0, 1.0);
        }

        return vec;
    };

    Vs::IOVector input = norm_dist_vec(10);
    Vs::IOVector output(10); output << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    auto objective_fn = Vs::SumOfSquaresObjective;
    double param_step_size = 0.00001;

    Vs::IOVector analytic_gradient = np->WeightGradient(input, output, objective_fn);
    Vs::IOVector numerical_gradient = Vs::CalculateNumericalGradient(np, input, output, objective_fn, param_step_size);

    std::cout << std::endl << "Top 10 Differences Of Analytic Vs Numerical" << std::endl;
    Vs::Util::TopTenDifferences(std::cout, analytic_gradient, numerical_gradient, [&np](size_t row, size_t col){ std::stringstream ss; np->DescribeParamIdx(ss, row); return ss.str();});

    GTEST_ASSERT_LE((analytic_gradient - numerical_gradient).norm(), 0.01);
}

TEST(Network, AnalyticAndNumericalGradientsMatchSoftmaxAndLogLoss) {
    auto np = std::make_shared<Vs::Network>(5);
    np->AddFullyConnectedLayer(5, Vs::ReLu);
    auto softmax_layer_idx = np->AddSoftMaxLayer();
    // np->SetAllWeightsTo(0.15);

    auto norm_dist_vec = [](size_t dim) {
        Vs::IOVector vec(dim);
        for (size_t idx = 0; idx < dim; idx++) {
            vec(idx) = Vs::Util::RandInGaussian(0.0, 1.0);
        }

        return vec;
    };

    Vs::IOVector input = norm_dist_vec(5); // ; input << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
    Vs::IOVector output(5); output << 1, 0, 0, 0, 0; // 0, 0, 0, 0, 0;
    auto objective_fn = Vs::LogLossObjective;
    double param_step_size = 0.00001;

    Vs::IOVector analytic_gradient = np->WeightGradient(input, output, objective_fn);
    Vs::IOVector numerical_gradient = Vs::CalculateNumericalGradient(np, input, output, objective_fn, param_step_size);

    std::cout << std::endl << "Softmax(ReLu) layer differences:" << std::endl;
    Vs::Util::DifferencesForIndicies(std::cout, np->ParamIndiciesForLayerWeights(softmax_layer_idx - 1), analytic_gradient, numerical_gradient, [&np](size_t row, size_t col){ std::stringstream ss; np->DescribeParamIdx(ss, row); return ss.str();});
    GTEST_ASSERT_LE((analytic_gradient - numerical_gradient).norm(), 0.01);
}

TEST(Network, AnalyticAndNumericalGradientsMatchSimpleSoftMax) {
    auto np = std::make_shared<Vs::Network>(2);
    np->AddFullyConnectedLayer(2, Vs::ReLu);
    auto softmax_layer_idx = np->AddSoftMaxLayer();
    // np->SetAllWeightsTo(0.15);

    auto norm_dist_vec = [](size_t dim) {
        Vs::IOVector vec(dim);
        for (size_t idx = 0; idx < dim; idx++) {
            vec(idx) = Vs::Util::RandInGaussian(0.0, 1.0);
        }

        return vec;
    };

    Vs::IOVector input = norm_dist_vec(2);
    Vs::IOVector output(2); output << 1, 0;
    auto objective_fn = Vs::SumOfSquaresObjective;
    double param_step_size = 0.00001;

    Vs::IOVector analytic_gradient = np->WeightGradient(input, output, objective_fn);
    Vs::IOVector numerical_gradient = Vs::CalculateNumericalGradient(np, input, output, objective_fn, param_step_size);

    std::cout << std::endl << "Softmax(ReLu) layer differences:" << std::endl;
    Vs::Util::DifferencesForIndicies(std::cout, np->ParamIndiciesForLayerWeights(softmax_layer_idx - 1), analytic_gradient, numerical_gradient, [&np](size_t row, size_t col){ std::stringstream ss; np->DescribeParamIdx(ss, row); return ss.str();});
    GTEST_ASSERT_LE((analytic_gradient - numerical_gradient).norm(), 0.01);

}
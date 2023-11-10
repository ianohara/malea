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
    n.SetUnityWeights();

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

        auto index_for = [&n](int row, int col) { return row * n.GetNodeCount() + col; };
        ASSERT_NEAR(gradient(index_for(3, 5)), del_E_del_w_35, NetEps);
        ASSERT_NEAR(gradient(index_for(1, 3)), del_E_del_w_13, NetEps);
    };

    n.SetUnityWeights();
    test(1, 0, 0);
    test(0, 0, 0);
    test(0.1, 0.2, 0.2);

    auto rand_weights = [](size_t row, size_t col) {
        return (std::rand() / static_cast<Vs::WVal>(RAND_MAX)) - 1.0;
    };
    n.SetWeightsWith(rand_weights);

    test(1, 0, 0);
}
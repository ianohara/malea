#include "gtest/gtest.h"

#include "network.hpp"

static constexpr Vs::FVal NetEps = 0.0001;

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

TEST(Network, TwoNodePassThroughLayers) {
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

TEST(Network, BigHonkerReLu) {
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

TEST(Network, GradientSingleNodeLayers) {
    const size_t num_layers = 4;
    Vs::Network n(1);

    for (auto layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        n.AddFullyConnectedLayer(1, Vs::ReLu);
    }
    n.SetUnityWeights();

    Vs::IOVector input(1);
    input.fill(1);

    auto out = n.WeightGradient(input, input, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientMultipleNodeMultipleLayer) {
    const size_t num_layers = 4;
    Vs::Network n(3);

    for (auto layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        n.AddFullyConnectedLayer(3, Vs::ReLu);
    }
    n.SetUnityWeights();

    Vs::IOVector input(3);
    input.fill(0.5);

    // Just make sure it runs!
    auto out = n.WeightGradient(input, 2*input, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientBigHonkin) {
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
    auto out = n.WeightGradient(input, 0.1*input, Vs::SumOfSquaresObjective);
}

TEST(Network, GradientHandCalcChecks) {
    Vs::Network n(1);
    n.AddFullyConnectedLayer(2, Vs::ArgCubed);
    n.AddFullyConnectedLayer(2, Vs::ArgCubed);
    n.AddFullyConnectedLayer(2, Vs::ArgCubed);
    n.SetUnityWeights();

    Vs::IOVector input(1);
    input << 1.0;
    Vs::IOVector output(2);
    output << 2.0, 3.0;

    auto node_outputs = n.Apply(input);
    auto gradient = n.WeightGradient(input, output, Vs::SumOfSquaresObjective);

    auto i_5 = n.GetWeightForConnection(3, 5) * node_outputs[3] + n.GetWeightForConnection(4, 5) * node_outputs[4];
    auto i_6 = n.GetWeightForConnection(3, 6) * node_outputs[3] + n.GetWeightForConnection(4, 6) * node_outputs[4];
    auto i_3 = n.GetWeightForConnection(1, 3) * node_outputs[1] + n.GetWeightForConnection(2, 3) * node_outputs[2];
    auto del_E_del_w_35 = -2.0 * (output[0] - node_outputs[5]) * 3.0 * i_5 * i_5 * node_outputs[3];

    auto del_E_del_w_13 = -2.0 * (output[0] - node_outputs[5]) * 3.0 * i_5 * i_5 * (n.GetWeightForConnection(3, 5) * 3.0 * i_3 * i_3 * node_outputs[1])
        - 2.0 * (output[1] - node_outputs[6])*(3.0 * i_6 * i_6) * (n.GetWeightForConnection(3, 6) * 3.0 * i_3 * i_3 * node_outputs[1]);

    auto index_for = [=](int row, int col) { return row * n.GetNodeCount() + col; };

    std::cout << gradient << std::endl;

    ASSERT_NEAR(gradient(index_for(1, 3)), del_E_del_w_13, NetEps);
    ASSERT_NEAR(gradient(index_for(3, 5)), del_E_del_w_35, NetEps);

}
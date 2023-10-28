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

    ASSERT_NEAR(n.Apply(input).sum(), 101.1, NetEps);
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

    ASSERT_NEAR(n.Apply(input).sum(), 0, NetEps);

    input << 2.0;
    ASSERT_NEAR(n.Apply(input).sum(), 2.0, NetEps);
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
    ASSERT_NEAR(n.Apply(input).sum(), 32, NetEps);
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

    ASSERT_NEAR(n.Apply(input).sum(), honkin_size * 123.0, NetEps);
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
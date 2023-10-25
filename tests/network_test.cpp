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
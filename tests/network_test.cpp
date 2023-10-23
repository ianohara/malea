#include "gtest/gtest.h"

#include "network.hpp"

TEST(Network, Create)
{
    Vs::Network n(200);
}

TEST(Network, FullyConnected)
{
    Vs::Network n(5);
    n.AddFullyConnectedLayer(3, Vs::ReLu());
    n.AddFullyConnectedLayer(4, Vs::ReLu());
}
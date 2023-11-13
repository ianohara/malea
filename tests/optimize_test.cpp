#include "gtest/gtest.h"

#include "optimize.hpp"

TEST(Optimize, BasicWorking) {
    Vs::AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8, 200);

    Vs::ParamVector current(200);
    Vs::ParamVector gradient = Vs::ParamVector::Random(200);
    current.fill(0);

    current = optimizer.Step(current, gradient);
}
#include "gtest/gtest.h"

#include "optimize.hpp"

TEST(Optimize, BasicWorking) {
    Vs::AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8, 200);

    Vs::ParamVector current(200);
    Vs::ParamVector gradient = Vs::ParamVector::Random(200);
    current.fill(0);

    current = optimizer.Step(current, gradient);
}

TEST(Optimize, RastriginFunction) {
    // See https://en.wikipedia.org/wiki/Rastrigin_function
    constexpr int dims = 9;
    constexpr double A = 10;
    Vs::AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8, dims);

    Vs::ParamVector current = Vs::ParamVector::Random(dims) * 5.12;
    Vs::ParamVector gradient(dims);
    gradient.fill(0);

    auto rastrigin_apply = [&](Vs::ParamVector args) -> double {
        return A * dims + (args.array()*args.array() - A*Eigen::cos(M_2_PI * args.array())).sum();
    };

    auto rastragin_gradient = [&](Vs::ParamVector args) -> Vs::ParamVector {
        Vs::ParamVector gradient = 2 * args.array() + M_2_PI * A * Eigen::sin(M_2_PI * args.array());
        return gradient;
    };

    while (current.sum() > 0.1) {
        std::cout << "Current is: " << current.transpose() << std::endl;
        current = optimizer.Step(current, rastragin_gradient(current));
    }
}
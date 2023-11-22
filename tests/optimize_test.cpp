#include "optimize.hpp"

#include "gtest/gtest.h"
#include "util.hpp"

TEST(Optimize, BasicWorking) {
    Vs::AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8, 200);

    Vs::ParamVector current(200);
    Vs::ParamVector gradient = Vs::ParamVector::Random(200);
    current.fill(0);

    current = optimizer.Step(current, gradient);
}

TEST(Optimize, DISABLED_RastriginFunction) {
    // See https://en.wikipedia.org/wiki/Rastrigin_function
    constexpr int dims = 9;
    constexpr double A = 10;
    Vs::AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8, dims);

    // The suggested search domain is -5.12 to 5.12 for all the dimensions.  Set most of them
    // pretty far away from the minimum (at 0) but set a few to random within that range.
    Vs::ParamVector current(dims);
    current.fill(-1);
    current(3) = 2;
    current(8) = 0.5;

    Vs::ParamVector gradient(dims);
    gradient.fill(0);

    auto rastrigin_apply = [&](Vs::ParamVector args) -> double {
        return A * dims + (args.array() * args.array() - A * Eigen::cos(2 * M_PI * args.array())).sum();
    };

    auto rastragin_gradient = [&](Vs::ParamVector args) -> Vs::ParamVector {
        Vs::ParamVector gradient = 2 * args.array() + 2 * M_PI * A * Eigen::sin(2 * M_PI * args.array());
        return gradient;
    };

    size_t step_count = 0;
    while (current.norm() > 0.01) {
        Vs::ParamVector gradient = rastragin_gradient(current);
        current = optimizer.Step(current, gradient);
        step_count++;
        GTEST_ASSERT_TRUE(step_count < 2000) << "The optimizer should converge in less than 2000 (arbitrary) steps on "
                                                "the 9 dimensional Rastrigin Function.";
    }
}
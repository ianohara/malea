#include <iostream>
#include <memory>
#include <string>

#include "util.hpp"

namespace Vs {
class ActivationFunction {
   public:
    virtual FVal Apply(size_t node_idx, const IOVector& node_inputs) = 0;
    virtual Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) = 0;
    virtual std::string Describe() = 0;
};

class _ReLuImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override {
        return node_inputs(node_idx) <= 0.0 ? 0.0 : node_inputs(node_idx);
    }

    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override {
        Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> J(node_inputs.size(), node_inputs.size());
        J.fill(0);

        for (ssize_t idx = 0; idx < node_inputs.size(); idx++) {
            J(idx, idx) = node_inputs(idx) <= 0.0 ? 0.0 : 1;
        }

        return J;
    }

    std::string Describe() override { return "ReLuActivation"; }
};

class _PassThroughImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override { return node_inputs(node_idx); }

    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override {
        return Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic>::Zero(node_inputs.size(), node_inputs.size());
    }

    std::string Describe() override { return "PassThroughActivation"; }
};

class _ArgCubedImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override { return std::pow(node_inputs(node_idx), 3); }

    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override {
        Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> J(node_inputs.size(), node_inputs.size());
        J.fill(0);
        for (ssize_t idx = 0; idx < node_inputs.size(); idx++) {
            J(idx, idx) = 3.0 * std::pow(node_inputs(idx), 2);
        }

        return J;
    }

    std::string Describe() override { return "ArgCubedActivation"; }
};

class _SigmoidImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override {
        return 1.0 / (1.0 - std::exp(-node_inputs(node_idx)));
    }

    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override {
        Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> J(node_inputs.size(), node_inputs.size());
        J.fill(0);
        for (ssize_t idx = 0; idx < node_inputs.size(); idx++) {
            auto apply = Apply(idx, node_inputs);
            J(idx, idx) = apply * (1.0 - apply);
        }

        return J;
    }

    std::string Describe() override { return "Sigmoid"; }
};

class _SoftMaxImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override {
        // To avoid overflow, use the softmax(x) = softmax(x + c) identity where
        // c is some constant.
        //
        // You can prove this by using the fact that any y^(a+b) = y^a * y^b and the
        // fact that you can factor constants out of summations.
        FVal node_input_max = node_inputs.maxCoeff();
        const auto e = std::exp(1);
        const auto numerator = Eigen::pow(e, node_inputs.array() - node_input_max);
        FVal denom = numerator.sum();
        return numerator(node_idx) / denom;
    }

    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override {
        IOVector partial_derivs(node_inputs.size());
        IOVector all_applied(node_inputs.size());
        for (ssize_t idx = 0; idx < node_inputs.size(); idx++) {
            all_applied(idx) = Apply(idx, node_inputs);
        }

        Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> J(node_inputs.size(), node_inputs.size());

        for (ssize_t idx_i = 0; idx_i < node_inputs.size(); idx_i++) {
            for (ssize_t idx_j = 0; idx_j < node_inputs.rows(); idx_j++) {
                if (idx_i == idx_j) {
                    // Use idx and node_idx in this expression (even though equal!) just to denote that
                    // we're doing a partial derivative wrt node_idx, and this is the partial when the
                    // component index is the same as node_idx.
                    //
                    // There are a bunch of articles on this, or you can just do it by hand, but I think
                    // I like this article best:
                    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
                    J(idx_i, idx_j) = all_applied(idx_i) * (1 - all_applied(idx_j));
                } else {
                    J(idx_i, idx_j) = -all_applied(idx_i) * all_applied(idx_j);
                }
            }
        }

        return J;
    }

    std::string Describe() override { return "SoftMaxActivation"; }
};

class ObjectiveFunction {
   public:
    virtual FVal Apply(IOVector final_layer_output, IOVector expected_output) = 0;
    virtual IOVector Gradient(IOVector final_layer_output, IOVector expected_output) = 0;
    virtual std::string Describe() = 0;
};

class _SumOfSquares : public ObjectiveFunction {
   public:
    FVal Apply(IOVector final_layer_output, IOVector expected_output) override {
        auto differences = expected_output - final_layer_output;
        return differences.squaredNorm();
    }

    // This is the derivative of the ObjectiveFunction with respect to each of the final layer
    // outputs.  Since the expected outputs and outputs are constant, this is just the sum of the elements of
    // 2*(expected-final_layer) * -1 since the derivative of the argument wrt the final layer outputs is -1
    // This only returns the given requested index.  EG the derivative wrt last layer node id final_layer_idx.
    IOVector Gradient(IOVector final_layer_output, IOVector expected_output) override {
        IOVector differences = -2.0 * (expected_output - final_layer_output);
        return differences;
    }

    std::string Describe() override { return "SumOfSquaresObjective"; }
};

class _LogLoss : public ObjectiveFunction {
   public:
    FVal Apply(IOVector final_layer_output, IOVector expected_output) override {
        if (Vs::Debug) {
            // For log loss, the expected output should be a "one hot" vector (1 entry is 1.0, all others 0.0) so
            // exact comparisons work below.
            assert(expected_output.nonZeros() == 1);
            assert(expected_output.array().sum() == 1.0);
        }

        // The log loss function is sum(expected*log(actual)) for each entry.  Since only 1 expected output is non-zero,
        // we could find that one and multiply versus its actual entry (aka ignore the sum) but just in case, just do
        // the full calc here.
        return (-expected_output.array() * Eigen::log(final_layer_output.array())).sum();
    }

    IOVector Gradient(IOVector final_layer_output, IOVector expected_output) override {
        IOVector all_vals = (-expected_output.array() * (1 / final_layer_output.array()));
        return all_vals;
    }

    std::string Describe() override { return "LogLossObjective"; }
};

static auto SumOfSquaresObjective = std::make_shared<_SumOfSquares>();
static auto LogLossObjective = std::make_shared<_LogLoss>();

static auto ReLu = std::make_shared<_ReLuImpl>();
static auto PassThrough = std::make_shared<_PassThroughImpl>();
static auto ArgCubed = std::make_shared<_ArgCubedImpl>();  // Used for testing
static auto SoftMax = std::make_shared<_SoftMaxImpl>();
static auto Sigmoid = std::make_shared<_SigmoidImpl>();
}  // namespace Vs
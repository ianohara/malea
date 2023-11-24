#ifndef NETWORK_FUNCTIONS_HPP
#define NETWORK_FUNCTIONS_HPP
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
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override;
    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override;
    std::string Describe() override;
};

class _PassThroughImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override;
    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override;
    std::string Describe() override;
};

class _ArgCubedImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override;
    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override;
    std::string Describe() override;
};

class _SigmoidImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override;
    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override;
    std::string Describe() override;
};

class _SoftMaxImpl : public ActivationFunction {
   public:
    FVal Apply(size_t node_idx, const IOVector& node_inputs) override;
    Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> Jacobian(const IOVector& node_inputs) override;
    std::string Describe() override;
};

class ObjectiveFunction {
   public:
    virtual FVal Apply(IOVector final_layer_output, IOVector expected_output) = 0;
    virtual IOVector Gradient(IOVector final_layer_output, IOVector expected_output) = 0;
    virtual std::string Describe() = 0;
};

class _SumOfSquares : public ObjectiveFunction {
   public:
    FVal Apply(IOVector final_layer_output, IOVector expected_output) override;
    // This is the derivative of the ObjectiveFunction with respect to each of the final layer
    // outputs.  Since the expected outputs and outputs are constant, this is just the sum of the elements of
    // 2*(expected-final_layer) * -1 since the derivative of the argument wrt the final layer outputs is -1
    // This only returns the given requested index.  EG the derivative wrt last layer node id final_layer_idx.
    IOVector Gradient(IOVector final_layer_output, IOVector expected_output) override;
    std::string Describe() override;
};

class _LogLoss : public ObjectiveFunction {
    public:
    FVal Apply(IOVector final_layer_output, IOVector expected_output) override;
    IOVector Gradient(IOVector final_layer_output, IOVector expected_output) override;
    std::string Describe() override;
};

static auto SumOfSquaresObjective = std::make_shared<_SumOfSquares>();
static auto LogLossObjective = std::make_shared<_LogLoss>();

static auto ReLu = std::make_shared<_ReLuImpl>();
static auto PassThrough = std::make_shared<_PassThroughImpl>();
static auto ArgCubed = std::make_shared<_ArgCubedImpl>();  // Used for testing
static auto SoftMax = std::make_shared<_SoftMaxImpl>();
static auto Sigmoid = std::make_shared<_SigmoidImpl>();
}  // namespace Vs
#endif /* NETWORK_FUNCTIONS_HPP */
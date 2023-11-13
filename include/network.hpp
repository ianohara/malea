#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <iostream>
#include <string>

#include "Eigen/Core"

#include "util.hpp"

namespace Vs {
    static constexpr bool Debug = false;

    typedef double FVal;
    typedef double BVal;
    typedef double WVal;

    typedef Eigen::Matrix<FVal, Eigen::Dynamic, 1> IOVector;
    typedef Eigen::Matrix<BVal, Eigen::Dynamic, 1> GradVector;

    class ActivationFunction {
    public:
        virtual FVal Apply(size_t node_idx, const IOVector& node_inputs) = 0;
        virtual BVal Derivative(size_t node_idx, const IOVector& node_inputs) = 0;
        virtual std::string Describe() = 0;
    };

    class _ReLuImpl : public ActivationFunction {
    public:
        FVal Apply(size_t node_idx, const IOVector& node_inputs) override {
            return node_inputs(node_idx) <= 0.0 ? 0.0 : node_inputs(node_idx);
        }

        BVal Derivative(size_t node_idx, const IOVector& node_inputs) override {
            return node_inputs(node_idx) <= 0.0 ? 0.0 : 1.0;
        }

        std::string Describe() override {
            return "ReLuActivation";
        }
    };

    class _PassThroughImpl : public ActivationFunction {
    public:
        FVal Apply(size_t node_idx, const IOVector& node_inputs) override {
            return node_inputs(node_idx);
        }

        BVal Derivative(size_t node_idx, const IOVector& node_inputs) override {
            return 0.0;
        }

        std::string Describe() override {
            return "PassThroughActivation";
        }
    };

    class _ArgCubedImpl : public ActivationFunction {
    public:
        FVal Apply(size_t node_idx, const IOVector& node_inputs) override {
            return std::pow(node_inputs(node_idx), 3);
        }

        BVal Derivative(size_t node_idx, const IOVector& node_inputs) override {
            return 3.0 * std::pow(node_inputs(node_idx), 2);
        }

        std::string Describe() override {
            return "ArgCubedActivation";
        }
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

        BVal Derivative(size_t node_idx, const IOVector& node_inputs) override {
            IOVector partial_derivs(node_inputs.rows());

            for (size_t idx = 0; idx < node_inputs.rows(); idx++) {
                if (idx == node_idx) {
                    // Use idx and node_idx in this expression (even though equal!) just to denote that
                    // we're doing a partial derivative wrt node_idx, and this is the partial when the
                    // component index is the same as node_idx.
                    //
                    // There are a bunch of articles on this, or you can just do it by hand, but I think
                    // I like this article best: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
                    partial_derivs(idx) = node_inputs(idx) * (1 - node_inputs(node_idx));
                } else {
                    partial_derivs(idx) = -node_inputs(idx) * node_inputs(node_idx);
                }
            }

            // NOTE(imo): I don't think this is right, but am just plowing forward for now.  This is calculating
            // the row of the Jacobian of the softmax layer where the derivative of the nth row in the softmax
            // (our node_idx) is taken with respect to every other node input.  Then to get the "derivative"
            // for node_idx we multiply each of those by the incoming node value for each node.  Don't think this
            // is right, so will need to debug this.
            return (partial_derivs.transpose() * node_inputs)(0);
        }

        std::string Describe() override {
            return "SoftMaxActivation";
        }
    };

    class ObjectiveFunction {
    public:
        virtual FVal Apply(IOVector final_layer_output, IOVector expected_output) = 0;
        virtual BVal Derivative(IOVector final_layer_output, IOVector expected_output, size_t final_layer_idx) = 0;
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
        BVal Derivative(IOVector final_layer_output, IOVector expected_output, size_t final_layer_idx) override {
            auto differences = -2.0 * (expected_output - final_layer_output);
            return differences(final_layer_idx);
        }

        std::string Describe() override {
            return "SumOfSquaresObjective";
        }
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

            // The log loss function is sum(expected*log(actual)) for each entry.  Since only 1 expected output is non-zero, we could
            // find that one and multiply versus its actual entry (aka ignore the sum) but just in case, just do the full calc here.
            return (-expected_output.array() * Eigen::log(final_layer_output.array())).sum();
        }

        BVal Derivative(IOVector final_layer_output, IOVector expected_output, size_t final_layer_idx) override {
            return (-expected_output.array() * (1 / final_layer_output.array()))(final_layer_idx);
        }

        std::string Describe() override {
            return "LogLossObjective";
        }
    };

    static auto SumOfSquaresObjective = std::make_shared<_SumOfSquares>();
    static auto LogLossObjective = std::make_shared<_LogLoss>();

    static auto ReLu = std::make_shared<_ReLuImpl>();
    static auto PassThrough = std::make_shared<_PassThroughImpl>();
    static auto ArgCubed = std::make_shared<_ArgCubedImpl>();  // Used for testing
    static auto SoftMax = std::make_shared<_SoftMaxImpl>();

    class Network {
    public:
        Network(size_t input_size) : weights(input_size, input_size), connections(input_size, input_size), layer_nodes { std::make_pair(0, input_size - 1) }, activation_functions { PassThrough } {
            weights.fill(0.0);
            connections.fill(0.0);
        }

        void AddLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn);

        // Adds a layer that is fully connected to the previously added layer.
        void AddFullyConnectedLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn);

        void AddSoftMaxLayer();

        inline size_t GetLayerForNode(size_t node_idx) {
            for (size_t layer_idx = 0; layer_idx < layer_nodes.size(); layer_idx++) {
                auto layer_pair = layer_nodes[layer_idx];
                if (node_idx >= layer_pair.first && node_idx <= layer_pair.second) {
                    return layer_idx;
                }
            }

            throw;
        }

        inline std::vector<size_t> GetNodesForLayer(size_t layer_idx) {
            auto[start_idx, end_idx] = layer_nodes[layer_idx];
            std::vector<size_t> ids;
            for (auto id = start_idx; id <= end_idx; id++) {
                ids.push_back(id);
            }

            return ids;
        }

        inline std::vector<size_t> GetIncomingNodesFor(size_t node_idx) {
            std::vector<size_t> ids;
            for (size_t row_idx = 0; row_idx < connections.rows(); row_idx++) {
                if (connections(row_idx, node_idx)) {
                    ids.push_back(row_idx);
                }
            }

            return ids;
        }

        inline std::vector<size_t> GetOutgoingNodesFor(size_t node_idx) {
            std::vector<size_t> ids;
            for (size_t col_idx = 0; col_idx < connections.cols(); col_idx++) {
                if (connections(node_idx, col_idx)) {
                    ids.push_back(col_idx);
                }
            }

            return ids;
        }

        inline size_t GetNodeCount() const {
            // Nodes are 0 indexed, and layer_nodes has inclusive pairs, so the count is + 1
            return layer_nodes.back().second + 1;
        }

        inline size_t GetLayerNodeCount(size_t layer_idx) {
            auto [from, to] = layer_nodes[layer_idx];
            return to - from + 1;
        }

        inline FVal GetWeightForConnection(size_t from, size_t to) {
            return weights(from, to);
        }

        inline void SetWeightForConnection(size_t from_idx, size_t to_idx, WVal weight) {
            weights(from_idx, to_idx) = weight;
        }

        inline bool GetAreConnected(size_t from_idx, size_t to_idx) {
            return connections(from_idx, to_idx);
        }

        inline void SetConnected(size_t from_idx, size_t to_idx, bool connected) {
            connections(from_idx, to_idx) = connected;
        }

        inline size_t GetLayerCount() {
            return layer_nodes.size();
        }

        IOVector Apply(IOVector input);

        GradVector WeightGradient(IOVector input, IOVector output, std::shared_ptr<ObjectiveFunction> objective_fn);

        inline void SetAllWeightsTo(WVal weight) {
            weights.fill(weight);
        }

        inline void SetUnityWeights() {
            SetAllWeightsTo(1.0);
        }

        inline void SetWeightsWith(std::function<WVal(size_t, size_t)> weight_setter) {
            for (size_t row = 0; row < connections.rows(); row++) {
                for (size_t col = 0; col < connections.cols(); col++) {
                    if (connections(row, col)) {
                        weights(row, col) = weight_setter(row, col);
                    }
                }
            }
        }

        IOVector OutputVectorFromNodeOutputs(IOVector apply_output) {
            auto last_layer_nodes = GetNodesForLayer(GetLayerCount() - 1);
            IOVector output(last_layer_nodes.size());
            for (size_t idx = 0; idx < last_layer_nodes.size(); idx++) {
                output(idx) = apply_output(last_layer_nodes[idx]);
            }

            return output;
        }

        void HeInitialize(size_t layer_idx);

        size_t GetOptimizedWeightCount() {
            return weights.cols() * weights.rows();
        }

        void SetOptimizedWeights(Eigen::Matrix<WVal, Eigen::Dynamic, 1> new_weights) {
            assert(weights.size() == new_weights.size());
            // TODO(imo): Make this more efficient...
            size_t count = 0;
            for (size_t col = 0; col < weights.cols(); col++) {
                for (size_t row = 0; row < weights.rows(); row++) {
                    weights(row, col) = new_weights(count++);
                }
            }
        }

    private:
        // Each row represents the "from" node, and each column represents the "to" node.  So
        // entry [i][j] is the weight for the connection from node i to node j in the network.
        Eigen::Matrix<WVal, Eigen::Dynamic, Eigen::Dynamic> weights;

        // Each row represents the "from" node, and each column represents the "to" node.  So
        // entry [i][j] shows whether node i is connected to node j (in that direction).  The
        // "from" side is the "closer to the input of the network" node.
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> connections;

        // This keeps track of which nodes are in which layer.  The pairs represent the starting
        // and ending nodes of the layer (inclusive).  Eg, layer_nodes[5] = (200, 203) means that
        // layer 5 consists of nodes 200, 201, 202, and 203.
        //
        // For now the only assumption is that layer 0 is the input layer, and the last layer
        // represents the output vector.
        std::vector<std::pair<size_t, size_t>> layer_nodes;

        // For each layer, the activation function used for its nodes.
        std::vector<std::shared_ptr<ActivationFunction>> activation_functions;

        // The scratch pad for keeping track of the forward evaluation values of nodes.  This is only
        // valid for a particular input vector, and really shouldn't be used outside of Apply and
        // WeightGradient
        // TODO(imo): Get rid of this
        IOVector node_output_values;

        // The scratch pad for keeping track of the derivative of a loss function with respect to a
        // node's input for each node as the backpropagation process is carried out.
        std::vector<BVal> del_objective_del_node_input;

        void ResizeForNodeCount(size_t old_node_count, size_t new_node_count);

        FVal ApplyNode(size_t node_idx, const IOVector& node_values);

        // This gets all the inputs for a layer's nodes (the weighted sum of incoming connections)
        // In order for it to work, the node_output_values for the incoming node connections
        // must be up to date!
        IOVector GetLayerInputs(size_t layer_id) {
            auto node_ids = GetNodesForLayer(layer_id);
            IOVector inputs(node_ids.size());
            size_t input_idx = 0;
            for (size_t node_id : node_ids) {
                auto incoming_node_ids = GetIncomingNodesFor(node_id);
                FVal accumulated_input = 0;
                for (size_t in_node_id : incoming_node_ids) {
                    accumulated_input += GetWeightForConnection(in_node_id, node_id) * node_output_values[in_node_id];
                }
                inputs(input_idx) = accumulated_input;
                input_idx++;
            }

            return inputs;
        }

        size_t GlobalNodeToLayerNode(size_t global_node_id) {
            size_t layer_idx = GetLayerForNode(global_node_id);
            size_t starting_idx = layer_nodes[layer_idx].first;

            // If the starting ID is 11, and global_node_id is 11, we should get 0 since it's
            // the 0th node in the layer.
            return global_node_id - starting_idx;
        }
    };
}

#endif
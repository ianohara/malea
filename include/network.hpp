#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <iostream>
#include <string>

#include "Eigen/Core"

namespace Vs {
    typedef double FVal;
    typedef double BVal;
    typedef double WVal;

    typedef Eigen::Matrix<FVal, Eigen::Dynamic, 1> IOVector;
    typedef Eigen::Matrix<BVal, Eigen::Dynamic, 1> GradVector;

    class ActivationFunction {
    public:
        virtual FVal Apply(FVal in) = 0;
        virtual BVal Derivative(FVal in) = 0;
        virtual std::string Describe() = 0;
    private:
    };

    class _ReLuImpl : public ActivationFunction {
    public:
        FVal Apply(FVal in) override {
            return in <= 0.0 ? 0.0 : in;
        }

        BVal Derivative(FVal in) override {
            return in <= 0.0 ? 0.0 : 1.0;
        }

        std::string Describe() override {
            return "ReLu";
        }
    };

    class _PassThroughImpl : public ActivationFunction {
    public:
        FVal Apply(FVal in) override {
            return in;
        }

        BVal Derivative(FVal in) override {
            return 0.0;
        }

        std::string Describe() override {
            return "PassThrough";
        }
    };

    static auto ReLu = std::make_shared<_ReLuImpl>();
    static auto PassThrough = std::make_shared<_PassThroughImpl>();

    class Network {
    public:
        Network(size_t input_size) : weights(input_size, input_size), connections(input_size, input_size), layer_nodes { std::make_pair(0, input_size - 1) }, activation_functions { PassThrough } {
            weights.fill(0.0);
            connections.fill(0.0);
        }

        void AddLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn);

        // Adds a layer that is fully connected to the previously added layer.
        void AddFullyConnectedLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn);

        inline size_t GetLayerForNode(size_t node_idx) {
            for (size_t layer_idx = 0; layer_idx < layer_nodes.size(); layer_idx++) {
                auto layer_pair = layer_nodes[layer_idx];
                if (node_idx >= layer_pair.first && node_idx <= layer_pair.second) {
                    return layer_idx;
                }
            }

            throw;
        }

        inline size_t GetNodeCount() {
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

        GradVector WeightGradient(IOVector input, IOVector output);

        inline void SetAllWeightsTo(WVal weight) {
            weights.fill(weight);
        }

        inline void SetUnityWeights() {
            SetAllWeightsTo(1.0);
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
        std::vector<FVal> node_output_values;

        void ResizeForNodeCount(size_t old_node_count, size_t new_node_count);

        FVal ApplyNode(size_t node_idx, FVal input);
    };
}

#endif
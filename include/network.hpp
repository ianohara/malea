#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <tuple>
#include <vector>

#include "Eigen/Core"

namespace Vs {
    typedef double FVal;
    typedef double BVal;
    typedef double WVal;

    typedef Eigen::Matrix<FVal, Eigen::Dynamic, 1> IOVector;
    typedef Eigen::Matrix<BVal, Eigen::Dynamic, 1> GradVector;

    class ActivationFunction {
    public:
        virtual FVal Apply(FVal in);
        virtual BVal Derivative(FVal in);
    private:
    };

    class ReLu : public ActivationFunction {
    public:
        FVal Apply(FVal in) override {
            return in <= 0.0 ? 0.0 : in;
        }

        BVal Derivative(FVal in) override {
            return in <= 0.0 ? 0.0 : 1.0;
        }
    };

    class PassThrough : public ActivationFunction {
    public:
        FVal Apply(FVal in) override {
            return in;
        }

        BVal Derivative(FVal in) override {
            return 0.0;
        }
    };

    class Network {
    public:
        Network(size_t input_size) : weights(input_size, input_size), connections(input_size, input_size), layer_nodes { std::make_pair(0, input_size - 1) }, activation_functions { PassThrough() } {}

        void AddLayer(size_t nodes, ActivationFunction fn);

        // Adds a layer that is fully connected to the previously added layer.
        void AddFullyConnectedLayer(size_t nodes, ActivationFunction fn);

        size_t GetNodeCount() {
            return layer_nodes.back().second;
        }

        size_t GetLayerNodeCount(size_t layer_idx) {
            auto [from, to] = layer_nodes[layer_idx];
            return to - from + 1;
        }

        FVal GetWeightForConnection(size_t from, size_t to) {
            return weights(from, to);
        }

        IOVector Apply(IOVector input);

        GradVector WeightGradient(IOVector input, IOVector output);

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
        std::vector<ActivationFunction> activation_functions;

        // The scratch pad for keeping track of the forward evaluation values of nodes.  This is only
        // valid for a particular input vector, and really shouldn't be used outside of Apply and
        // WeightGradient
        std::vector<FVal> node_output_values;

        void ResizeForAdditionalNodes(size_t new_nodes);

        void ResizeForNodeCount(size_t node_count);

        FVal ApplyNode(size_t node_idx, FVal input);
    };
}

#endif
#include "network.hpp"

#include <iostream>

namespace Vs {
    void Network::AddLayer(size_t nodes, ActivationFunction fn) {
        auto[from_start, from_end] = layer_nodes.back();

        layer_nodes.emplace_back(from_end + 1, from_end + nodes);
        auto[to_start, to_end] = layer_nodes.back();

        activation_functions.push_back(fn);

        ResizeForAdditionalNodes(nodes);
    }

    void Network::AddFullyConnectedLayer(size_t nodes, ActivationFunction fn) {
        auto[from_start, from_end] = layer_nodes.back();
        AddLayer(nodes, fn);
        auto[to_start, to_end] = layer_nodes.back();

        for (size_t m = from_start; m <= from_end; m++) {
            for (size_t n = to_start; n <= to_end; n++) {
                std::cout << "Adding connection from " << m << " to " << n << std::endl;
                connections(m, n) = true;
            }
        }
    }

    void Network::ResizeForAdditionalNodes(size_t new_nodes) {
        ResizeForNodeCount(GetNodeCount() + new_nodes);
    }

    void Network::ResizeForNodeCount(size_t node_count) {
        connections.resize(node_count, node_count);
        weights.resize(node_count, node_count);
        node_output_values.resize(node_count);
    }

    IOVector Network::Apply(IOVector input) {
        assert(input.rows() == GetLayerNodeCount(0));

        for (size_t n = 0; n < GetLayerNodeCount(0); n++) {
            FVal val = ApplyNode(n, input[n]);
            node_output_values[n] = val;
        }

        for (size_t n = GetLayerNodeCount(0); n < GetNodeCount(); n++) {
            FVal accumulated_input = 0.0;
            auto incoming_nodes = connections.col(n);
            // TODO(imo): change to sparse and use nonZeros
            for (size_t in_idx = 0; in_idx < incoming_nodes.rows(); in_idx++) {
                if (!incoming_nodes(in_idx)) {
                    continue;
                }

                auto weight = GetWeightForConnection(in_idx, n);
                accumulated_input += weight * node_output_values[in_idx];
            }

            node_output_values[n] = ApplyNode(n, accumulated_input);
        }

        auto[out_from, out_to] = layer_nodes.back();
        size_t out_len = out_to - out_from + 1;
        IOVector out_vec(out_len);

        for (size_t idx = 0; idx < out_len; idx++) {
            out_vec(idx) = node_output_values[out_from + idx];
        }

        return out_vec;
    }

    FVal Network::ApplyNode(size_t node_idx, FVal input) {
        return activation_functions[node_idx].Apply(input);
    }
}
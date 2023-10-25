#include "network.hpp"

#include <iostream>

namespace Vs {
    void Network::AddLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
        auto[from_start, from_end] = layer_nodes.back();

        layer_nodes.emplace_back(from_end + 1, from_end + nodes);
        auto[to_start, to_end] = layer_nodes.back();

        // TODO(imo: remove std::cout << "Just added (to_start, to_end)=(" << to_start << ", " << to_end << ") and (from_start, from_end)=(" << from_start << "," << from_end << ")" << std::endl;

        activation_functions.push_back(fn);
        auto old_node_count = from_end + 1;
        auto new_node_count = to_end + 1;

        ResizeForNodeCount(old_node_count, new_node_count);
        // TODO(imo: remove std::cout << "  GetNodeCount now=" << GetNodeCount() << std::endl;
    }

    void Network::AddFullyConnectedLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
        auto[from_start, from_end] = layer_nodes.back();
        AddLayer(nodes, fn);
        auto[to_start, to_end] = layer_nodes.back();

        for (size_t m = from_start; m <= from_end; m++) {
            for (size_t n = to_start; n <= to_end; n++) {
                // TODO(imo): remove std::cout << "Adding connection from " << m << " to " << n << std::endl;
                SetConnected(m, n, true);
            }
        }
    }

    void Network::ResizeForNodeCount(size_t old_count, size_t new_count) {
        // TODO(imo: remove std::cout << "old_count=" << old_count << " new_count=" << new_count << std::endl;
        auto new_first_index = old_count;
        auto added_nodes = new_count - old_count;

        // Make sure to preserve the existing values, and set the new to zero in both
        // connections and weights.  node_output_values doesn't need the same
        // treatment because it's just a scratch pad that is overwritten
        connections.conservativeResize(new_count, new_count);
        connections.block(new_first_index, 0, added_nodes, new_count).fill(0);
        connections.block(0, new_first_index, new_count, added_nodes).fill(0);

        weights.conservativeResize(new_count, new_count);
        weights.block(new_first_index, 0, added_nodes, new_count).fill(0);
        weights.block(0, new_first_index, new_count, added_nodes).fill(0);

        node_output_values.resize(new_count);
    }

    IOVector Network::Apply(IOVector input) {
        assert(input.rows() == GetLayerNodeCount(0));

        for (size_t n = 0; n < GetLayerNodeCount(0); n++) {
            FVal val = ApplyNode(n, input[n]);
            node_output_values[n] = val;
        }

        //std::cout << "in Apply, connections=" << std::endl << connections << std::endl;

        for (size_t n = GetLayerNodeCount(0); n < GetNodeCount(); n++) {
            FVal accumulated_input = 0.0;
            auto incoming_nodes = connections.col(n);
            // TODO(imo: remove std::cout << "for n=" << n << " incoming_nodes=" << std::endl << incoming_nodes << std::endl;
            // TODO(imo): change to sparse and use nonZeros
            for (size_t in_idx = 0; in_idx < incoming_nodes.rows(); in_idx++) {

                if (!incoming_nodes(in_idx)) {
                    // TODO(imo: remove std::cout << "    No connection, skipping" << std::endl;
                    continue;
                }
                //std::cout << "in_idx -> n : " << in_idx << " -> " << n << std::endl;
                auto weight = GetWeightForConnection(in_idx, n);
                accumulated_input += weight * node_output_values[in_idx];
                //std::cout << "    weight * node_output_values[in_idx] = " << weight * node_output_values[in_idx] << std::endl;
                // TODO(imo: removestd::cout << "    accumulated_input = " << accumulated_input << std::endl;
            }

            node_output_values[n] = ApplyNode(n, accumulated_input);
            //for (auto v : node_output_values)
            //    std::cout << v << " ";
            //std::cout << std::endl;
            //std::cout << "node_output_values[" << n << "] = " << node_output_values[n] << std::endl;
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
        auto layer_idx = GetLayerForNode(node_idx);
        auto fn = activation_functions[layer_idx];
        return fn->Apply(input);
    }
}
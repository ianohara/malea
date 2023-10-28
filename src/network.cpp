#include "network.hpp"

#include <iostream>

namespace Vs {
    void Network::AddLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
        auto[from_start, from_end] = layer_nodes.back();

        layer_nodes.emplace_back(from_end + 1, from_end + nodes);
        auto[to_start, to_end] = layer_nodes.back();

        activation_functions.push_back(fn);
        auto old_node_count = from_end + 1;
        auto new_node_count = to_end + 1;

        ResizeForNodeCount(old_node_count, new_node_count);
    }

    void Network::AddFullyConnectedLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
        auto[from_start, from_end] = layer_nodes.back();
        AddLayer(nodes, fn);
        auto[to_start, to_end] = layer_nodes.back();

        for (size_t m = from_start; m <= from_end; m++) {
            for (size_t n = to_start; n <= to_end; n++) {
                SetConnected(m, n, true);
            }
        }
    }

    void Network::ResizeForNodeCount(size_t old_count, size_t new_count) {
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
        del_objective_del_node_input.resize(new_count);
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
        auto layer_idx = GetLayerForNode(node_idx);
        auto fn = activation_functions[layer_idx];
        return fn->Apply(input);
    }

    GradVector Network::WeightGradient(IOVector input, IOVector expected_output, std::shared_ptr<ObjectiveFunction> objective_fn) {
        // NOTE(imo): We don't use the "applied_output" for grabbing most of the node output values below.
        // Instead, we use node_output_values which keeps track of the outputs of all the nodes (whereas
        // Apply just returns the output vector of the last layer).  We do use applied_output for evaluation
        // of the ObjectiveFunction, though.
        IOVector applied_output = Apply(input);

        // To calculate the gradient with respect to the weights, work backward
        // from the objective function in reverse node order.  For each node,
        // calculate the rate of change of its output with respect to each
        // of its incoming connection weights.  aka if each node has activation
        // function f_a(si) and si = sum(in_i * w_i) where in_i and w_i are the
        // incoming connection value and weight for each incomming connection,
        // and o_n = f_a(s_i) where o_n is the output value of the node then this
        // calculates del(o_n) / del(w_i) (the partial derivative of the output
        // with respect to each of its incomming weights).  To do this, it uses
        // the chain rule:
        //    del(o_n) / del(w_i) = del(o_n)/del(s_i) * del(si)/del(w_i)
        // Which for the case of si being a sum, is:
        //   del(o_n) / del(w_i) = del(o_n)/del(s_i) * w_i
        //
        // For the gradient of the objective function O(w_mn, I, E) (where
        // w_mn is the weight for connection from node m to n, I is a given input, E is
        // the expected output for the given input, and O is the ojbective function)
        // with respect to the weights, we have:
        //   O(w_mn, I, E) = O(o_l, I, E) where o_l is the output of the last layer
        //   del(O) / del(w_mn) = del(O) / del(o_l) * del(o_l) / del(w_mn)
        // And for weights leading into the last layer, this is calculable.  For
        // weights in the 2nd to last layer, o_l-1, the calculation is:
        //  del(O) / del(w_mn) = del(O) / del(o_l) * del(o_l) / del(o_l-1) * del(o_l-1)/del(w_mn)
        // where we need to be careful to expand to all the nodes and connections involved.  In
        // other words, each node of o_l might be connected to the node in o_l-1 into which w_mn
        // leads, and so the rate of change of O with respect to w_mn is affected by all of these.
        //
        // del(O) / del(w_mn) = sum(del(O) / del(o_li)) for i nodes in last layer.
        //
        // The wikipedia article for this has a clearer (maybe?) explanation and a way of
        // representing this recursively.  The key is essentially that we need to calculate
        // the derivative of each node's output with respect to its argument, which is easy
        // to do.  Then we collect these as we walk the node graph from objective to input,
        // and as a result can calculate the derivative with respect to each of the weights
        // without needing to do more than one pass over the node graph.
        //
        // Temporary variables needed:
        //    For each node, the derivative of the objective function with respect to its input argument
        // Process:
        //    For each node in the last layer store: del(O) / del(o_l) * del(o_l) / del(i_l)
        //      (the rate of change of the objective function with respect to the node's input)
        //    For each node in the last layer, del(O) / del(w_ji) is just:
        //      del(O) / del(o_l) * del(o_l) / del(i_l) * w_ji
        //    For each node in the l-1 layer, calculate del(o_l-1i) / del(i_l-1i)
        //      (the rate of change of the node with respect to the node's input)
        //    and then:
        //      del(O) / del(w_ji) = sum(w_ji * del(O)/del(i_i))
        //    So we want to store the del(O)/del(i_i) values so that each i-1 layer
        //    can refer to them.  Node that i-2 needs i-1 and i layers, but if we
        //    go in reverse like this, then both will be available.
        //
        //
        // NOTE: The wikipedia article on backpropagation in the adjoint graph case,
        //  found [here](https://en.wikipedia.org/wiki/Backpropagation#Adjoint_graph),
        //  is probably a lot more clear and without errors!

        // The weight gradient vector is oriented the same as concatenating the rows
        // of the weight matrix.  This is so that the weights for the connections originating
        // from nodes earlier in the network come earlier in the gradient (just a convention,
        // doesn't matter).
        //
        // TODO(imo): Seems like this might end up being expensive and will need to be moved
        // somewhere that isn't reallocated.
        auto weight_gradient = Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic>();
        weight_gradient.resizeLike(weights);
        weight_gradient.fill(0);

        // TODO(imo): Remove this zeroing; it is just for debug.
        std::fill(del_objective_del_node_input.begin(), del_objective_del_node_input.end(), 0.0);

        // In the last layer, we special case for the derivative of the objective function wrt
        // its input arg (versus just the activation function).
        auto last_layer_idx = GetLayerCount() - 1;
        for (auto end_node_idx : GetNodesForLayer(last_layer_idx)) {
            del_objective_del_node_input[end_node_idx]
                = objective_fn->Derivative(applied_output, expected_output)
                    * activation_functions[last_layer_idx]->Derivative(node_output_values[end_node_idx]);
            for (auto incoming_node_idx : GetIncomingNodesFor(end_node_idx)) {
                weight_gradient(incoming_node_idx, end_node_idx) = node_output_values[incoming_node_idx] * GetWeightForConnection(incoming_node_idx, end_node_idx) * del_objective_del_node_input[end_node_idx];
            }
        }

        for (int layer_id = last_layer_idx - 1; layer_id >= 0; layer_id--) {
            for (auto current_node : GetNodesForLayer(layer_id)) {
                BVal outgoing_deriv_sum = 0.0;
                for (auto outgoing_node_idx : GetOutgoingNodesFor(current_node)) {
                    outgoing_deriv_sum += GetWeightForConnection(current_node, outgoing_node_idx) * del_objective_del_node_input[outgoing_node_idx];
                }

                del_objective_del_node_input[current_node]
                    = activation_functions[layer_id]->Derivative(node_output_values[current_node]) * outgoing_deriv_sum;

                for (auto incoming_node_idx : GetIncomingNodesFor(current_node)) {
                    weight_gradient(incoming_node_idx, current_node) = node_output_values[incoming_node_idx] * GetWeightForConnection(incoming_node_idx, current_node) * del_objective_del_node_input[current_node];
                }
            }
        }

        // Transpose rows to columns, then reshape to column vector.  This results in the desired vector with entries in order of increasing
        // connection "from" node id
        weight_gradient.transposeInPlace();
        return weight_gradient.reshaped(Eigen::AutoSize, 1);
    }
}
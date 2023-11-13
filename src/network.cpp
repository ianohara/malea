#include "network.hpp"

#include <iostream>

namespace Vs {
    void Network::AddLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
        // NOTE(imo): This assert is just a protection against using this in the
        // constructor for layer 0 at some point in the future.  Biases would be
        // broken by doing so.
        assert(layer_nodes.size() >= 1);

        auto[from_start, from_end] = layer_nodes.back();

        layer_nodes.emplace_back(from_end + 1, from_end + nodes);
        auto[to_start, to_end] = layer_nodes.back();

        activation_functions.push_back(fn);
        auto old_node_count = from_end + 1;
        auto new_node_count = to_end + 1;

        biases.push_back(0);

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

    void Network::AddSoftMaxLayer() {
        auto[from_start, from_end] = layer_nodes.back();
        AddLayer(from_end - from_start + 1, Vs::SoftMax);
        auto [to_start, to_end] = layer_nodes.back();

        assert((from_end - from_start) == (to_end - to_start));
        size_t from = from_start;
        size_t to = to_start;
        while (from <= from_end) {
            SetConnected(from, to, true);
            from++;
            to++;
        }
    }

    void Network::HeInitialize(size_t layer_idx) {
        for (size_t node_idx : GetNodesForLayer(layer_idx)) {
            const double incoming_count = GetIncomingNodesFor(node_idx).size();
            for (size_t incoming_idx : GetIncomingNodesFor(node_idx)) {
                double standard_dev = std::sqrt(2.0 / incoming_count);
                SetWeightForConnection(incoming_idx, node_idx, Vs::Util::RandInGaussian(0.0, standard_dev));
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

        for (size_t layer_idx = 0; layer_idx < GetLayerCount(); layer_idx++) {
            IOVector layer_inputs = layer_idx == 0 ? input : GetLayerInputs(layer_idx);
            for (size_t node_idx : GetNodesForLayer(layer_idx)) {
                node_output_values[node_idx] = ApplyNode(node_idx, layer_inputs);
            }
        }

        return node_output_values;
    }

    FVal Network::ApplyNode(size_t node_idx, const IOVector& node_values) {
        auto layer_idx = GetLayerForNode(node_idx);
        auto layer_node_idx = GlobalNodeToLayerNode(node_idx);
        auto fn = activation_functions[layer_idx];
        auto apply_value = fn->Apply(layer_node_idx, node_values);
        if (Vs::Debug) {
            std::cout << "Applying Node " << node_idx << ":" << std::endl
                << "  layer=" << layer_idx << std::endl
                << "  global=" << node_idx << "(layer_node=" << layer_node_idx << ")" << std::endl
                << "  node_values=" << node_values.transpose() << std::endl
                << "  apply_value=" << apply_value << std::endl;
        }
        return apply_value;
    }

    GradVector Network::WeightGradient(IOVector input, IOVector expected_output, std::shared_ptr<ObjectiveFunction> objective_fn) {
        // NOTE(imo): We don't use the "applied_output" for grabbing most of the node output values below.
        // Instead, we use node_output_values which keeps track of the outputs of all the nodes (whereas
        // Apply just returns the output vector of the last layer).  We do use applied_output for evaluation
        // of the ObjectiveFunction, though.
        IOVector node_outputs = Apply(input);
        IOVector applied_output = OutputVectorFromNodeOutputs(node_output_values);

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

        auto vec_string = [](IOVector vec) { return vec.transpose(); };

        // In the last layer, we special case for the derivative of the objective function wrt
        // its input arg (versus just the activation function).
        const size_t last_layer_idx = GetLayerCount() - 1;

        for (int layer_id = last_layer_idx; layer_id >= 0; layer_id--) {
            IOVector layer_inputs = GetLayerInputs(layer_id);

            for (auto current_node : GetNodesForLayer(layer_id)) {
                size_t node_idx_in_layer = GlobalNodeToLayerNode(current_node);
                const FVal activation_fn_deriv = activation_functions[layer_id]->Derivative(node_idx_in_layer, layer_inputs);
                if (layer_id == last_layer_idx) {
                    const FVal objective_fn_deriv = objective_fn->Derivative(applied_output, expected_output, node_idx_in_layer);
                    del_objective_del_node_input[current_node] = activation_fn_deriv * objective_fn_deriv;

                    if (Vs::Debug) {
                        std::cout << "For end node " << current_node << std::endl
                            << "   objective_fn derivative" << std::endl
                            << "     applied_output " << vec_string(applied_output) << std::endl
                            << "     expected_output " << vec_string(expected_output) << std::endl
                            << "     derivative " << objective_fn_deriv << std::endl
                            << "   incoming_weighted_sum " << layer_inputs[node_idx_in_layer] << std::endl
                            << "   activation_functions[last_layer_idx]->Derivative(incoming_weighted_sum) " << activation_fn_deriv << std::endl
                            << "   del_objective_del_node_input " << del_objective_del_node_input[current_node] << std::endl;
                    }
                } else {
                    BVal outgoing_deriv_sum = 0.0;
                    for (auto outgoing_node_idx : GetOutgoingNodesFor(current_node)) {
                        outgoing_deriv_sum += GetWeightForConnection(current_node, outgoing_node_idx) * del_objective_del_node_input[outgoing_node_idx];
                    }

                    if (Vs::Debug) {
                        std::cout << "For node " << current_node << " outoing_deriv_sum " << outgoing_deriv_sum << std::endl;
                    }

                    del_objective_del_node_input[current_node] = activation_fn_deriv * outgoing_deriv_sum;
                }

                // Each incoming connection has a weight associated with it, so fill in the weight_gradients for those edges since the rest of the partial
                // derivates have been calculated up to this point.
                for (auto incoming_node_idx : GetIncomingNodesFor(current_node)) {
                    weight_gradient(incoming_node_idx, current_node) = node_outputs[incoming_node_idx] * del_objective_del_node_input[current_node];
                }
            }
        }

        for (auto node_idx = 0; node_idx < GetNodeCount(); node_idx++) {
            FVal incoming_weighted_sum = 0.0;
            for (auto incoming_node_idx : GetIncomingNodesFor(node_idx)) {
                incoming_weighted_sum += node_outputs[incoming_node_idx] * GetWeightForConnection(incoming_node_idx, node_idx);
            }

            if (Vs::Debug) {
                std::cout << "For node " << node_idx << std::endl
                    << "  del_objective_del_node_input " << del_objective_del_node_input[node_idx] << std::endl
                    << "  node_outputs " << node_outputs[node_idx] << std::endl
                    << "  incoming_weighted_sum " << incoming_weighted_sum << std::endl;
                for (auto incoming_idx : GetIncomingNodesFor(node_idx)) {
                    std::cout << "  weight " << incoming_idx << "->" << node_idx << " " << GetWeightForConnection(incoming_idx, node_idx) << std::endl;
                }
            }
        }

        return weight_gradient.reshaped(Eigen::AutoSize, 1);
    }
}
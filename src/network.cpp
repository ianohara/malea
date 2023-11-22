#include "network.hpp"

#include <iostream>

namespace Vs {
size_t Network::AddLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
    // NOTE(imo): This assert is just a protection against using this in the
    // constructor for layer 0 at some point in the future.  Biases would be
    // broken by doing so.
    assert(layer_nodes.size() >= 1);

    auto [from_start, from_end] = layer_nodes.back();

    layer_nodes.emplace_back(from_end + 1, from_end + nodes);
    auto [to_start, to_end] = layer_nodes.back();

    activation_functions.push_back(fn);
    auto old_node_count = from_end + 1;
    auto new_node_count = to_end + 1;

    // If you want a layer without a bias, call DeleteBias after AddLayer
    size_t new_layer_idx = layer_nodes.size() - 1;
    biases[new_layer_idx] = 0.0;

    ResizeForNodeCount(old_node_count, new_node_count);

    return new_layer_idx;
}

size_t Network::AddFullyConnectedLayer(size_t nodes, std::shared_ptr<ActivationFunction> fn) {
    auto [from_start, from_end] = layer_nodes.back();
    const size_t new_layer_idx = AddLayer(nodes, fn);
    auto [to_start, to_end] = layer_nodes.back();

    for (size_t m = from_start; m <= from_end; m++) {
        for (size_t n = to_start; n <= to_end; n++) {
            SetConnected(m, n, true);
        }
    }

    HeInitialize(new_layer_idx);

    return new_layer_idx;
}

size_t Network::AddSoftMaxLayer() {
    auto [from_start, from_end] = layer_nodes.back();
    const size_t new_layer_idx = AddLayer(from_end - from_start + 1, Vs::SoftMax);
    DeleteBias(new_layer_idx);
    auto [to_start, to_end] = layer_nodes.back();

    assert((from_end - from_start) == (to_end - to_start));
    size_t from = from_start;
    size_t to = to_start;
    while (from <= from_end) {
        SetConnected(from, to, true);
        SetNotOptimizedUnityConnection(from, to);
        from++;
        to++;
    }

    return new_layer_idx;
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
    size_t new_first_index = old_count;
    size_t last_new_index = new_count - 1;

    for (size_t new_idx = new_first_index; new_idx <= last_new_index; new_idx++) {
        weights[new_idx] = {};
        connections[new_idx] = {};
    }

    node_output_values.resize(new_count);
    del_objective_del_node_input.resize(new_count);
}

IOVector Network::Apply(IOVector input) {
    assert(input.rows() == GetLayerNodeCount(0));

    for (size_t layer_idx = 0; layer_idx < GetLayerCount(); layer_idx++) {
        IOVector layer_inputs = layer_idx == 0 ? input : GetLayerInputs(layer_idx);
        for (size_t node_idx : GetNodesForLayer(layer_idx)) {
            auto node_output_value = ApplyNode(node_idx, layer_inputs);
            node_output_values[node_idx] = node_output_value;
        }
    }

    return node_output_values;
}

FVal Network::ApplyNode(size_t node_idx, const IOVector &node_values) {
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
    assert(!std::isnan(apply_value));
    return apply_value;
}

GradVector Network::WeightGradient(IOVector input, IOVector expected_output,
                                   std::shared_ptr<ObjectiveFunction> objective_fn) {
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
    //      (the rate of change of the node output with respect to the node's input)
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
    auto weight_gradient = Eigen::Matrix<BVal, Eigen::Dynamic, 1>(GetOptimizedWeightCount());
    auto connection_to_weight_idx = GetConnectionToWeightParamIdx();
    weight_gradient.fill(0);

    auto bias_gradient = Eigen::Matrix<BVal, Eigen::Dynamic, 1>();
    bias_gradient.resize(biases.size());
    bias_gradient.fill(0);

    // TODO(imo): Remove this zeroing; it is just for debug.
    std::fill(del_objective_del_node_input.begin(), del_objective_del_node_input.end(), 0.0);

    auto vec_string = [](IOVector vec) { return vec.transpose(); };

    // In the last layer, we special case for the derivative of the objective function wrt
    // its input arg (versus just the activation function).
    const size_t last_layer_idx = GetLayerCount() - 1;

    // Not all layers have biases, so we need a counter to keep track of where in the bias gradient we are.  We
    // start from the last bias.
    size_t current_bias_gradient_idx = biases.size() - 1;

    // NOTE(imo): Layer 0 doesn't have any weights associated with it, so we can skip it and only
    //   go down to layer 1.
    for (int layer_id = last_layer_idx; layer_id >= 1; layer_id--) {
        IOVector layer_inputs = GetLayerInputs(layer_id);
        BVal bias_grad_accum = 0;

        // Jacobian of activations is: N x N where N is the number of nodes in the current layer_id layer
        const Eigen::Matrix<BVal, Eigen::Dynamic, Eigen::Dynamic> activation_jacobian =
            activation_functions[layer_id]->Jacobian(layer_inputs);

        IOVector previous_layer_derivs(activation_jacobian.cols());
        if (layer_id == last_layer_idx) {
            previous_layer_derivs = objective_fn->Gradient(applied_output, expected_output);
        } else {
            for (auto current_node : GetNodesForLayer(layer_id)) {
                const size_t node_idx_in_layer = GlobalNodeToLayerNode(current_node);
                BVal outgoing_deriv_sum = 0.0;
                for (auto outgoing_node_idx : GetOutgoingNodesFor(current_node)) {
                    outgoing_deriv_sum += GetWeightForConnection(current_node, outgoing_node_idx) *
                                          del_objective_del_node_input[outgoing_node_idx];
                }
                previous_layer_derivs(node_idx_in_layer) = outgoing_deriv_sum;
            }
        }

        const IOVector this_layer_derivs = activation_jacobian * previous_layer_derivs;

        // std::cout << "For  layer " << layer_id << ":" << std::endl
        //     << "activation_jacobian   : " << std::endl << activation_jacobian << std::endl
        //     << "previous_layer_derivs : " << std::endl << previous_layer_derivs << std::endl
        //     << "this_layer_derivs     : " << std::endl << this_layer_derivs << std::endl << std::endl;

        if (biases.count(layer_id)) {
            // For biases, the "incoming" node value is 1 since it looks like a connection from a node with constant
            // output and weight == bias. So the gradient is just the sum of the derivatives of all the nodes it is
            // connected to (all this layer's nodes)
            bias_gradient(current_bias_gradient_idx--) = this_layer_derivs.sum();
        }

        for (auto current_node : GetNodesForLayer(layer_id)) {
            const size_t node_idx_in_layer = GlobalNodeToLayerNode(current_node);
            // Subsequent layers (earlier in the network) need to know the derivative of the objective wrt each node in
            // this layer, so record those values here.
            del_objective_del_node_input[current_node] = this_layer_derivs[node_idx_in_layer];

            // Each incoming connection has a weight associated with it, so fill in the weight_gradients for those edges
            // since the rest of the partial derivates have been calculated up to this point.
            for (auto incoming_node_idx : GetIncomingNodesFor(current_node)) {
                const bool this_is_optimized = IsOptimizedWeightConnection(incoming_node_idx, current_node);
                if (this_is_optimized) {
                    size_t weight_gradient_idx = connection_to_weight_idx[incoming_node_idx][current_node];
                    auto this_node_input = node_outputs[incoming_node_idx];
                    auto this_node_deriv = this_layer_derivs(node_idx_in_layer);
                    weight_gradient(weight_gradient_idx) = this_node_input * this_node_deriv;
                }
            }
        }
    }

    Eigen::Matrix<BVal, Eigen::Dynamic, 1> gradient(weight_gradient.rows() + bias_gradient.rows());
    gradient << weight_gradient, bias_gradient;

    return gradient;
}

Vs::GradVector CalculateNumericalGradient(const std::shared_ptr<Vs::Network> network, const Vs::IOVector &input,
                                          const Vs::IOVector &expected_output,
                                          const std::shared_ptr<Vs::ObjectiveFunction> objective_fn,
                                          const double param_step_size) {
    assert(param_step_size > 0);
    const Vs::IOVector starting_params = network->GetOptimizedParams();
    // For each parameter dimension, the change in objective function associated with a small step of param_step_size in
    // that dimension.  The step is done centered on the current location. Aka: -param_step_size/2 to param_step_size/2
    Vs::IOVector param_delta(starting_params.size());

    auto set_one_coeff = [&starting_params](size_t coeff_idx, double val) {
        Vs::IOVector zeros_except_one(starting_params.size());
        zeros_except_one.fill(0);
        zeros_except_one(coeff_idx) = val;

        Vs::IOVector result = starting_params + zeros_except_one;
        return result;
    };

    for (size_t dim_idx = 0; dim_idx < starting_params.rows(); dim_idx++) {
        const Vs::IOVector from_params = set_one_coeff(dim_idx, -param_step_size);
        const Vs::IOVector to_params = set_one_coeff(dim_idx, param_step_size);

        network->SetOptimizedParams(from_params);
        const double from_obj_val =
            objective_fn->Apply(network->OutputVectorFromNodeOutputs(network->Apply(input)), expected_output);
        network->SetOptimizedParams(to_params);
        const double to_obj_val =
            objective_fn->Apply(network->OutputVectorFromNodeOutputs(network->Apply(input)), expected_output);
        // std::cout << "CalculateNumericalGradient: step_size=" << param_step_size << " param norms: " <<
        // (from_params-starting_params).norm() << " & " << (to_params-starting_params).norm() << " objective vals:
        // from=" << from_obj_val << " to=" << to_obj_val << "(diff=" << to_obj_val - from_obj_val << ")" << std::endl;

        param_delta(dim_idx) = to_obj_val - from_obj_val;
    }

    network->SetOptimizedParams(starting_params);

    return param_delta.array() / (2.0 * param_step_size);
}
}  // namespace Vs
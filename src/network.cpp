#include "network.hpp"

#include <fstream>
#include <iostream>

namespace Ml {
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
    const size_t new_layer_idx = AddLayer(from_end - from_start + 1, Ml::SoftMax);
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
            SetWeightForConnection(incoming_idx, node_idx, Ml::Util::RandInGaussian(0.0, standard_dev));
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
    assert(input.rows() == static_cast<ssize_t>(GetLayerNodeCount(0)));

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
    if (Ml::Debug) {
        std::cout << "Applying Node " << node_idx << ":" << std::endl
                  << "  layer=" << layer_idx << std::endl
                  << "  global=" << node_idx << "(layer_node=" << layer_node_idx << ")" << std::endl
                  << "  node_values=" << node_values.transpose() << std::endl
                  << "  apply_value=" << apply_value << std::endl;
    }
    assert(!std::isnan(apply_value));
    return apply_value;
}

void Network::EnsureTempSizes() {
    size_t weight_size = GetOptimizedWeightCount();
    size_t bias_size = GetOptimizedBiasCount();

    if (static_cast<size_t>(weight_gradient.size()) != weight_size) {
        weight_gradient.resize(weight_size);
    }

    if (static_cast<size_t>(bias_gradient.size()) != bias_size) {
        bias_gradient.resize(bias_size);
    }
}

GradVector Network::WeightGradient(const IOVector& input, const IOVector& expected_output,
                                   std::shared_ptr<ObjectiveFunction> objective_fn) {
    EnsureTempSizes();
    // NOTE(imo): We don't use the "applied_output" for grabbing most of the node output values below.
    // Instead, we use node_output_values which keeps track of the outputs of all the nodes (whereas
    // Apply just returns the output vector of the last layer).  We do use applied_output for evaluation
    // of the ObjectiveFunction, though.
    const IOVector node_outputs = Apply(input);
    const IOVector applied_output = OutputVectorFromNodeOutputs(node_output_values);

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
    auto connection_to_weight_idx = GetConnectionToWeightParamIdx();

    // In the last layer, we special case for the derivative of the objective function wrt
    // its input arg (versus just the activation function).
    const size_t last_layer_idx = GetLayerCount() - 1;

    // Not all layers have biases, so we need a counter to keep track of where in the bias gradient we are.  We
    // start from the last bias.
    size_t current_bias_gradient_idx = biases.size() - 1;

    // NOTE(imo): Layer 0 doesn't have any weights associated with it, so we can skip it and only
    //   go down to layer 1.
    for (size_t layer_id = last_layer_idx; layer_id >= 1; layer_id--) {
        IOVector layer_inputs = GetLayerInputs(layer_id);

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

Ml::GradVector CalculateNumericalGradient(const std::shared_ptr<Ml::Network> network, const Ml::IOVector &input,
                                          const Ml::IOVector &expected_output,
                                          const std::shared_ptr<Ml::ObjectiveFunction> objective_fn,
                                          const double param_step_size) {
    assert(param_step_size > 0);
    const Ml::IOVector starting_params = network->GetOptimizedParams();
    // For each parameter dimension, the change in objective function associated with a small step of param_step_size in
    // that dimension.  The step is done centered on the current location. Aka: -param_step_size/2 to param_step_size/2
    Ml::IOVector param_delta(starting_params.size());

    auto set_one_coeff = [&starting_params](size_t coeff_idx, double val) {
        Ml::IOVector zeros_except_one(starting_params.size());
        zeros_except_one.fill(0);
        zeros_except_one(coeff_idx) = val;

        Ml::IOVector result = starting_params + zeros_except_one;
        return result;
    };

    for (ssize_t dim_idx = 0; dim_idx < starting_params.rows(); dim_idx++) {
        const Ml::IOVector from_params = set_one_coeff(dim_idx, -param_step_size);
        const Ml::IOVector to_params = set_one_coeff(dim_idx, param_step_size);

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

void Network::SerializeTo(std::string out_file) {
    std::ofstream os(out_file, std::ios::binary);
    if (!os.good()) {
        std::stringstream ss;
        ss << "Could not open file for binary output: '" << out_file << "'";
        throw std::runtime_error(ss.str());
    }

    const IOVector params = GetOptimizedParams();
    const ssize_t param_count = params.size();
    os.seekp(0);
    os.write(reinterpret_cast<const char *>(&param_count), sizeof(param_count));
    for (ssize_t idx = 0; idx < param_count; idx++) {
        const double this_param = params(idx);
        os.write(reinterpret_cast<const char *>(&this_param), sizeof(this_param));
    }
}

void Network::DeserializeFrom(std::string in_file) {
    std::ifstream is(in_file, std::ios::binary);
    if (!is.good()) {
        std::stringstream ss;
        ss << "Could not open file for binary input: '" << in_file << "'";
        throw std::runtime_error(ss.str());
    }

    is.seekg(0);
    IOVector read_params(GetOptimizedParamsCount());
    ssize_t param_count_in_file = 0;
    is.read(reinterpret_cast<char *>(&param_count_in_file), sizeof(param_count_in_file));

    if (param_count_in_file != read_params.size()) {
        std::stringstream ss;
        ss << "Param count of network is " << read_params.size() << " but file contains " << param_count_in_file;
        throw std::runtime_error(ss.str());
    }

    for (ssize_t idx = 0; idx < param_count_in_file; idx++) {
        double this_param;
        is.read(reinterpret_cast<char *>(&this_param), sizeof(this_param));
        read_params(idx) = this_param;
    }

    SetOptimizedParams(read_params);
}

void Network::SetConnected(size_t from_idx, size_t to_idx, bool connected) {
        auto already_connected = GetAreConnected(from_idx, to_idx);
        auto from_idx_outgoing = GetOutgoingNodesFor(from_idx);
        if (already_connected) {
            if (connected) {
                return;
            } else {
                std::remove(from_idx_outgoing.begin(), from_idx_outgoing.end(), to_idx);
                DeleteWeight(from_idx, to_idx);
            }
        } else {
            if (connected) {
                connections[from_idx].push_back(to_idx);
                // Don't have a weight by default.  The connected but no weight
                // case is equivalent to unity weight that isn't optimized.
                DeleteWeight(from_idx, to_idx);
            } else {
                return;
            }
        }
    }

void Network::SetWeightsWith(std::function<WVal(size_t, size_t)> weight_setter) {
        for (auto [from_idx, to_weights] : weights) {
            for (auto [to_idx, current_weight] : to_weights) {
                assert(GetAreConnected(from_idx, to_idx));
                auto new_val = weight_setter(from_idx, to_idx);
                weights[from_idx][to_idx] = new_val;
            }
        }
    }

    IOVector Network::OutputVectorFromNodeOutputs(const IOVector& apply_output) {
        auto last_layer_nodes = GetNodesForLayer(GetLayerCount() - 1);
        IOVector output(last_layer_nodes.size());
        for (size_t idx = 0; idx < last_layer_nodes.size(); idx++) {
            output(idx) = apply_output(last_layer_nodes[idx]);
        }

        return output;
    }

    size_t Network::GetOptimizedWeightCount() {
        size_t weight_count = 0;
        for (auto [from_idx, to_map] : weights) {
            weight_count += to_map.size();
        }

        return weight_count;
    }

    IOVector Network:: GetOptimizedWeights() {
        IOVector weights_in_order(GetOptimizedWeightCount());
        size_t current_idx = 0;
        for (auto [from_idx, to_map] : weights) {
            for (auto [to_idx, weight] : to_map) {
                weights_in_order[current_idx++] = weight;
            }
        }

        return weights_in_order;
    }

    IOVector Network::GetOptimizedBiases() {
        IOVector biases_in_order(GetOptimizedBiasCount());

        size_t current_idx = 0;
        for (auto [layer_idx, bias] : biases) {
            biases_in_order[current_idx++] = bias;
        }

        return biases_in_order;
    }

    void Network::SetOptimizedParams(const IOVector& new_params) {
        assert(static_cast<ssize_t>(GetOptimizedParamsCount()) == new_params.size());
        // TODO(imo): Make this more efficient...
        size_t count = 0;
        for (auto [from_idx, to_map] : weights) {
            for (auto [to_idx, weight] : to_map) {
                SetWeightForConnection(from_idx, to_idx, new_params[count++]);
            }
        }

        for (auto [layer_idx, bias_val] : biases) {
            biases[layer_idx] = new_params(count++);
        }
    }

    void Network::DescribeParamIdx(std::ostream &stream, size_t param_idx) {
        size_t count = 0;
        for (auto [from_idx, to_map] : weights) {
            for (auto [to_idx, weight] : to_map) {
                if (count == param_idx) {
                    stream << "Weight " << from_idx << "->" << to_idx;
                    return;
                }
                count++;
            }
        }

        for (auto [layer_idx, bias] : biases) {
            if (count == param_idx) {
                stream << "Bias Layer " << layer_idx;
                return;
            }
            count++;
        }

        stream << "Invalid Param Idx " << param_idx;
    }

    std::vector<size_t> Network::ParamIndiciesForLayerWeights(const size_t layer_idx) {
        size_t count = 0;
        auto layer_node_idx = GetNodesForLayer(layer_idx);
        std::vector<size_t> param_indicies;
        for (auto [from_idx, to_map] : weights) {
            for (auto [to_idx, weight] : to_map) {
                if (std::find(layer_node_idx.begin(), layer_node_idx.end(), to_idx) != std::end(layer_node_idx)) {
                    param_indicies.push_back(count);
                }
                count++;
            }
        }

        return param_indicies;
    }

    void Network::SummarizeNonZeroParams(std::ostream &stream) {
        auto non_zero_count = [](const OrderedWeightMap &w) {
            size_t non_zero_count = 0;
            for (auto [from_idx, to_map] : w) {
                for (auto [to_idx, weight] : to_map) {
                    if (std::abs(weight) > 0.00001) {
                        non_zero_count++;
                    }
                }
            }

            return non_zero_count;
        };

        auto non_zero_bias_count = [](const Eigen::Matrix<FVal, Eigen::Dynamic, Eigen::Dynamic> m) {
            size_t non_zero_count = 0;
            for (ssize_t row = 0; row < m.rows(); row++) {
                for (ssize_t col = 0; col < m.cols(); col++) {
                    if (std::abs(m(row, col)) > 0.00001) {
                        non_zero_count++;
                    }
                }
            }
            return non_zero_count;
        };

        auto connection_count = [](const OrderedConnectionMap &c) {
            size_t non_zero_count = 0;
            for (auto [from_idx, to_idx_vec] : c) {
                non_zero_count += to_idx_vec.size();
            }

            return non_zero_count;
        };

        auto n_vals = [](Eigen::Matrix<FVal, Eigen::Dynamic, Eigen::Dynamic> m, size_t n, bool min) {
            std::vector<BVal> all_vals;
            for (ssize_t row = 0; row < m.rows(); row++) {
                for (ssize_t col = 0; col < m.cols(); col++) {
                    all_vals.push_back(m(row, col));
                }
            }

            if (min) {
                std::sort(all_vals.begin(), all_vals.end());
            } else {
                std::sort(all_vals.rbegin(), all_vals.rend());
            }

            std::stringstream ss;
            for (size_t idx = 0; idx < n && idx < all_vals.size(); idx++) {
                ss << all_vals[idx] << ",";
            }

            return ss.str();
        };

        stream << "Network Parameter Summary" << std::endl
               << "  Optimized Weight Count: " << GetOptimizedWeightCount() << std::endl
               << "  Non-zero weight count : " << non_zero_count(weights) << std::endl
               << "  Max 100 weights       : " << n_vals(GetOptimizedWeights(), 100, false) << std::endl
               << "  Min 100 weights       : " << n_vals(GetOptimizedWeights(), 100, true) << std::endl
               << "  Optimized Bias Count  : " << GetOptimizedParamsCount() - GetOptimizedWeightCount() << std::endl
               << "  Non-zero biases count : " << non_zero_bias_count(GetOptimizedBiases()) << std::endl
               << "  Max 100 biases        : " << n_vals(GetOptimizedBiases(), 100, false) << std::endl
               << "  Min 100 biases        : " << n_vals(GetOptimizedBiases(), 100, true) << std::endl
               << "  Bias Count            : " << biases.size() << std::endl
               << "  Connection count      : " << connection_count(connections) << std::endl;
    }

    void Network::SummarizeParamGradient(std::ostream &stream, const IOVector &gradient) {
        auto n_vals = [&gradient](size_t n, bool min) {
            std::vector<BVal> all_vals;
            for (ssize_t row = 0; row < gradient.rows(); row++) {
                all_vals.push_back(gradient(row));
            }

            if (min) {
                std::sort(all_vals.begin(), all_vals.end());
            } else {
                std::sort(all_vals.rbegin(), all_vals.rend());
            }

            std::stringstream ss;
            for (size_t idx = 0; idx < n; idx++) {
                ss << all_vals[idx] << ",";
            }

            return ss.str();
        };

        auto min_100 = [&gradient, &n_vals]() { return n_vals(100, true); };
        auto max_100 = [&gradient, &n_vals]() { return n_vals(100, false); };

        stream << "Gradient Summary" << std::endl
               << "  gradient rows,cols = " << gradient.rows() << ", " << gradient.cols() << std::endl
               << "  gradient min 10 vals " << min_100() << std::endl
               << "  gradient max 10 vals " << max_100() << std::endl;
    }

    void Network::SummarizeObjDelNode(std::ostream &stream) {
        stream << "Summary of objective gradient wrt each node" << std::endl;
        for (size_t layer_idx = 1; layer_idx < GetLayerCount(); layer_idx++) {
            auto node_inputs = GetLayerInputs(layer_idx);
            size_t node_input_idx = 0;
            stream << "  Layer " << layer_idx << std::endl;
            for (size_t node_idx : GetNodesForLayer(layer_idx)) {
                stream << "    n=" << node_idx << " del_O/del_i=" << del_objective_del_node_input[node_idx]
                       << " (node_input=" << node_inputs[node_input_idx++] << ")"
                       << " (node_output=" << node_output_values[node_idx] << ")" << std::endl;
            }
        }
    }

    void Network::SummarizeNodeOutputs(std::ostream &stream, const IOVector &input, bool include_input_layer) {
        auto node_outputs = Apply(input);
        stream << "Summary of node outputs" << std::endl;
        for (size_t layer_idx = include_input_layer ? 0 : 1; layer_idx < GetLayerCount(); layer_idx++) {
            size_t input_index = 0;
            stream << "  Layer " << layer_idx << std::endl;
            for (size_t node_idx : GetNodesForLayer(layer_idx)) {
                stream << "    n=" << node_idx << " output=" << node_outputs[node_idx];
                if (layer_idx > 0) {
                    auto layer_inputs = GetLayerInputs(layer_idx);
                    stream << " (input=" << layer_inputs[input_index++] << ")";
                }
                stream << std::endl;
            }
        }
    }

    void Network::SummarizeWeightsForLayer(std::ostream &stream, const size_t layer_idx,
                                  const size_t max_to_print /* 0 for all, this is per node */) {
        stream << "Weights & Bias for layer " << layer_idx << std::endl;
        if (biases.count(layer_idx)) {
            stream << "  bias=" << biases[layer_idx] << std::endl;
        } else {
            stream << "  bias=none (layer " << layer_idx << ")" << std::endl;
        }

        for (size_t node_idx : GetNodesForLayer(layer_idx)) {
            stream << "  node " << node_idx << std::endl;
            size_t this_count = 0;
            for (size_t incoming_idx : GetIncomingNodesFor(node_idx)) {
                if (max_to_print && this_count >= max_to_print) {
                    break;
                }
                stream << "    " << incoming_idx << "->" << node_idx << "="
                       << GetWeightForConnection(incoming_idx, node_idx) << std::endl;
                this_count++;
            }
        }
    }

    std::map<size_t, std::map<size_t, size_t>> Network::GetConnectionToWeightParamIdx() {
        size_t current_index = 0;
        std::map<size_t, std::map<size_t, size_t>> connection_to_idx;
        for (auto [from_idx, to_map] : weights) {
            connection_to_idx[from_idx] = {};
            for (auto [to_idx, weight] : to_map) {
                if (IsOptimizedWeightConnection(from_idx, to_idx)) {
                    connection_to_idx[from_idx][to_idx] = current_index++;
                }
            }
        }

        return connection_to_idx;
    }

    std::vector<size_t> Network::GetIncomingNodesFor(size_t node_idx) {
        std::vector<size_t> ids;
        const size_t to_layer_idx = GetLayerForNode(node_idx);
        // This is the input layer
        if (to_layer_idx == 0) {
            return ids;
        }

        const size_t from_layer_idx = to_layer_idx - 1;
        for (size_t from_node_idx : GetNodesForLayer(from_layer_idx)) {
            if (connections.count(from_node_idx)) {
                auto from_node_connections = connections.at(from_node_idx);
                if (std::find(from_node_connections.begin(), from_node_connections.end(), node_idx) !=
                    std::end(from_node_connections)) {
                    ids.push_back(from_node_idx);
                }
            }
        }

        return ids;
    }
}  // namespace Ml
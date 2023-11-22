#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "Eigen/Core"
#include "network_functions.hpp"
#include "util.hpp"

namespace Vs {
class Network {
   public:
    Network(size_t input_size) : layer_nodes{std::make_pair(0, input_size - 1)}, activation_functions{Vs::PassThrough} {
        ResizeForNodeCount(0, input_size);
    }

    size_t AddLayer(size_t nodes, std::shared_ptr<Vs::ActivationFunction> fn);

    // Adds a layer that is fully connected to the previously added layer.
    size_t AddFullyConnectedLayer(size_t nodes, std::shared_ptr<Vs::ActivationFunction> fn);

    size_t AddSoftMaxLayer();

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
        auto [start_idx, end_idx] = layer_nodes[layer_idx];
        std::vector<size_t> ids;
        for (auto id = start_idx; id <= end_idx; id++) {
            ids.push_back(id);
        }

        return ids;
    }

    inline std::vector<size_t> GetIncomingNodesFor(size_t node_idx) {
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

    inline std::vector<size_t> GetOutgoingNodesFor(size_t node_idx) { return connections.at(node_idx); }

    inline size_t GetNodeCount() const {
        // Nodes are 0 indexed, and layer_nodes has inclusive pairs, so the count is + 1
        return layer_nodes.back().second + 1;
    }

    inline size_t GetLayerNodeCount(size_t layer_idx) {
        auto [from, to] = layer_nodes[layer_idx];
        return to - from + 1;
    }

    inline FVal GetWeightForConnection(size_t from, size_t to) {
        assert(GetAreConnected(from, to));
        if (weights.count(from) && weights[from].count(to)) {
            return weights[from][to];
        } else {
            return 1.0;
        }
    }

    inline void SetNotOptimizedUnityConnection(size_t from, size_t to) {
        assert(GetAreConnected(from, to));
        DeleteWeight(from, to);
    }

    inline void SetWeightForConnection(size_t from_idx, size_t to_idx, WVal weight) {
        weights[from_idx][to_idx] = weight;
    }

    inline bool GetAreConnected(size_t from_idx, size_t to_idx) {
        if (!connections.count(from_idx)) {
            return false;
        }

        auto outgoing_connections = connections.at(from_idx);
        return (std::find(outgoing_connections.begin(), outgoing_connections.end(), to_idx) !=
                std::end(outgoing_connections));
    }

    inline void SetBiasForLayer(size_t layer_idx, WVal bias) { biases[layer_idx] = bias; }

    inline void DeleteBias(size_t layer_idx) { biases.erase(layer_idx); }

    inline WVal GetBias(size_t layer_idx) { return biases.at(layer_idx); }

    inline void SetConnected(size_t from_idx, size_t to_idx, bool connected) {
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

    inline size_t GetLayerCount() { return layer_nodes.size(); }

    IOVector Apply(IOVector input);

    GradVector WeightGradient(IOVector input, IOVector output, std::shared_ptr<Vs::ObjectiveFunction> objective_fn);

    inline void SetAllWeightsTo(WVal weight) {
        auto a_constant = [weight](size_t row, size_t col) { return weight; };
        SetWeightsWith(a_constant);
    }

    inline void SetUnityWeights() { SetAllWeightsTo(1.0); }

    inline void SetWeightsWith(std::function<WVal(size_t, size_t)> weight_setter) {
        for (auto [from_idx, to_weights] : weights) {
            for (auto [to_idx, current_weight] : to_weights) {
                assert(GetAreConnected(from_idx, to_idx));
                auto new_val = weight_setter(from_idx, to_idx);
                weights[from_idx][to_idx] = new_val;
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

    size_t GetOptimizedParamsCount() { return GetOptimizedWeightCount() + GetOptimizedBiasCount(); }

    size_t GetOptimizedWeightCount() {
        size_t weight_count = 0;
        for (auto [from_idx, to_map] : weights) {
            weight_count += to_map.size();
        }

        return weight_count;
    }

    size_t GetOptimizedBiasCount() { return biases.size(); }

    IOVector GetOptimizedWeights() {
        IOVector weights_in_order(GetOptimizedWeightCount());
        size_t current_idx = 0;
        for (auto [from_idx, to_map] : weights) {
            for (auto [to_idx, weight] : to_map) {
                weights_in_order[current_idx++] = weight;
            }
        }

        return weights_in_order;
    }

    IOVector GetOptimizedBiases() {
        IOVector biases_in_order(GetOptimizedBiasCount());

        size_t current_idx = 0;
        for (auto [layer_idx, bias] : biases) {
            biases_in_order[current_idx++] = bias;
        }

        return biases_in_order;
    }

    IOVector GetOptimizedParams() {
        IOVector params(GetOptimizedParamsCount());

        params << GetOptimizedWeights(), GetOptimizedBiases();
        return params;
    }

    void SetOptimizedParams(Eigen::Matrix<WVal, Eigen::Dynamic, 1> new_params) {
        assert((GetOptimizedParamsCount()) == new_params.size());
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

    void DescribeParamIdx(std::ostream &stream, size_t param_idx) {
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

    std::vector<size_t> ParamIndiciesForLayerWeights(const size_t layer_idx) {
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

    void SummarizeNonZeroParams(std::ostream &stream) {
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
            for (size_t row = 0; row < m.rows(); row++) {
                for (size_t col = 0; col < m.cols(); col++) {
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
            for (size_t row = 0; row < m.rows(); row++) {
                for (size_t col = 0; col < m.cols(); col++) {
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

    void SummarizeParamGradient(std::ostream &stream, const IOVector &gradient) {
        auto n_vals = [&gradient](size_t n, bool min) {
            std::vector<BVal> all_vals;
            for (size_t row = 0; row < gradient.rows(); row++) {
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

    // NOTE(imo): This uses the internal results of the last call to WeightGradient.  It's invalid to call
    // if WeightGradient has not been called, or if Apply has been called subsequently!
    void SummarizeObjDelNode(std::ostream &stream) {
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

    // Given this input vector, apply it and then print out all the network's node outputs layer by layer.
    void SummarizeNodeOutputs(std::ostream &stream, const IOVector &input, bool include_input_layer) {
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

    void SummarizeWeightsForLayer(std::ostream &stream, const size_t layer_idx,
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

    void SummarizeAllForwardDebugInfo(std::ostream &stream, const IOVector &input, const bool include_input_layer) {
        SummarizeNonZeroParams(stream);
        SummarizeNodeOutputs(stream, input, include_input_layer);
        for (size_t layer_idx = 0; layer_idx < GetLayerCount(); layer_idx++) {
            stream << "Layer " << layer_idx << " activation function: " << activation_functions[layer_idx]->Describe()
                   << std::endl;
            SummarizeWeightsForLayer(stream, layer_idx, 0);
        }
    }

    std::map<size_t, std::map<size_t, size_t>> GetConnectionToWeightParamIdx() {
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

    inline bool IsOptimizedWeightConnection(size_t from_idx, size_t to_idx) {
        return weights.count(from_idx) && weights[from_idx].count(to_idx);
    }

   private:
    typedef std::map<size_t, std::map<size_t, WVal>> OrderedWeightMap;
    typedef std::map<size_t, std::vector<size_t>> OrderedConnectionMap;

    // Each row represents the "from" node, and each column represents the "to" node.  So
    // entry [i][j] is the weight for the connection from node i to node j in the network.
    // Eigen::Matrix<WVal, Eigen::Dynamic, Eigen::Dynamic> weights;
    OrderedWeightMap weights;

    // Each row represents the "from" node, and each column represents the "to" node.  So
    // entry [i][j] shows whether node i is connected to node j (in that direction).  The
    // "from" side is the "closer to the input of the network" node.
    // Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> connections;
    OrderedConnectionMap connections;

    // This keeps track of which nodes are in which layer.  The pairs represent the starting
    // and ending nodes of the layer (inclusive).  Eg, layer_nodes[5] = (200, 203) means that
    // layer 5 consists of nodes 200, 201, 202, and 203.
    //
    // For now the only assumption is that layer 0 is the input layer, and the last layer
    // represents the output vector.
    std::vector<std::pair<size_t, size_t>> layer_nodes;

    // For each layer, the activation function used for its nodes.
    std::vector<std::shared_ptr<Vs::ActivationFunction>> activation_functions;

    // Each layer has a constant bias that is added to its node's inputs.
    //
    // NOTE(imo): layer 0 does not have a bias, and some layers like softmax do not either.
    std::map<size_t, Vs::FVal> biases;

    // The scratch pad for keeping track of the forward evaluation values of nodes.  This is only
    // valid for a particular input vector, and really shouldn't be used outside of Apply and
    // WeightGradient
    // TODO(imo): Get rid of this
    IOVector node_output_values;

    // The scratch pad for keeping track of the derivative of a loss function with respect to a
    // node's input for each node as the backpropagation process is carried out.
    std::vector<BVal> del_objective_del_node_input;

    void ResizeForNodeCount(size_t old_node_count, size_t new_node_count);

    FVal ApplyNode(size_t node_idx, const IOVector &node_values);

    // This gets all the inputs for a layer's nodes (the weighted sum of incoming connections)
    // In order for it to work, the node_output_values for the incoming node connections
    // must be up to date!
    //
    // NOTE(imo): Getting layer inputs for layer 0 is invalid since it is the input layer.
    IOVector GetLayerInputs(size_t layer_id) {
        assert(layer_id >= 1 && layer_id < layer_nodes.size());
        auto node_ids = GetNodesForLayer(layer_id);
        IOVector inputs(node_ids.size());
        size_t input_idx = 0;
        for (size_t node_id : node_ids) {
            auto incoming_node_ids = GetIncomingNodesFor(node_id);
            FVal accumulated_input = 0;
            for (size_t in_node_id : incoming_node_ids) {
                accumulated_input += GetWeightForConnection(in_node_id, node_id) * node_output_values[in_node_id];
            }
            if (biases.count(layer_id)) {
                accumulated_input += biases[layer_id];
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

    void DeleteWeight(size_t from_idx, size_t to_idx) {
        if (weights.count(from_idx)) {
            weights.at(from_idx).erase(to_idx);
        }
    }
};

Vs::GradVector CalculateNumericalGradient(const std::shared_ptr<Vs::Network> network, const Vs::IOVector &input,
                                          const Vs::IOVector &expected_output,
                                          const std::shared_ptr<Vs::ObjectiveFunction> objective_fn,
                                          const double param_step_size);
}  // namespace Vs

#endif
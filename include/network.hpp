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

    std::vector<size_t> GetIncomingNodesFor(size_t node_idx);

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

    void SetConnected(size_t from_idx, size_t to_idx, bool connected);

    inline size_t GetLayerCount() { return layer_nodes.size(); }

    IOVector Apply(IOVector input);

    GradVector WeightGradient(const IOVector& input, const IOVector& output, std::shared_ptr<Vs::ObjectiveFunction> objective_fn);

    inline void SetAllWeightsTo(WVal weight) {
        auto a_constant = [weight](size_t row, size_t col) { return weight; };
        SetWeightsWith(a_constant);
    }

    inline void SetUnityWeights() { SetAllWeightsTo(1.0); }

    void SetWeightsWith(std::function<WVal(size_t, size_t)> weight_setter);

    IOVector OutputVectorFromNodeOutputs(const IOVector& apply_output);

    void HeInitialize(size_t layer_idx);

    size_t GetOptimizedParamsCount() { return GetOptimizedWeightCount() + GetOptimizedBiasCount(); }

    size_t GetOptimizedWeightCount();

    size_t GetOptimizedBiasCount() { return biases.size(); }

    IOVector GetOptimizedWeights();

    IOVector GetOptimizedBiases();

    IOVector GetOptimizedParams() {
        IOVector params(GetOptimizedParamsCount());

        params << GetOptimizedWeights(), GetOptimizedBiases();
        return params;
    }

    void SetOptimizedParams(const IOVector& new_params);

    void DescribeParamIdx(std::ostream &stream, size_t param_idx);

    std::vector<size_t> ParamIndiciesForLayerWeights(const size_t layer_idx);

    void SummarizeNonZeroParams(std::ostream &stream);

    void SummarizeParamGradient(std::ostream &stream, const IOVector &gradient);

    // NOTE(imo): This uses the internal results of the last call to WeightGradient.  It's invalid to call
    // if WeightGradient has not been called, or if Apply has been called subsequently!
    void SummarizeObjDelNode(std::ostream &stream);

    // Given this input vector, apply it and then print out all the network's node outputs layer by layer.
    void SummarizeNodeOutputs(std::ostream &stream, const IOVector &input, bool include_input_layer);

    void SummarizeWeightsForLayer(std::ostream &stream, const size_t layer_idx,
                                  const size_t max_to_print /* 0 for all, this is per node */);

    void SummarizeAllForwardDebugInfo(std::ostream &stream, const IOVector &input, const bool include_input_layer) {
        SummarizeNonZeroParams(stream);
        SummarizeNodeOutputs(stream, input, include_input_layer);
        for (size_t layer_idx = 0; layer_idx < GetLayerCount(); layer_idx++) {
            stream << "Layer " << layer_idx << " activation function: " << activation_functions[layer_idx]->Describe()
                   << std::endl;
            SummarizeWeightsForLayer(stream, layer_idx, 0);
        }
    }

    std::map<size_t, std::map<size_t, size_t>> GetConnectionToWeightParamIdx();

    inline bool IsOptimizedWeightConnection(size_t from_idx, size_t to_idx) {
        return weights.count(from_idx) && weights[from_idx].count(to_idx);
    }

    // For now, these just serialize and deserialize the weights and biases and not the network structure.
    // So you'll need to make sure that the param vector these load from file match your network's!
    void SerializeTo(std::string out_file);

    void DeserializeFrom(std::string in_file);

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
    IOVector weight_gradient;
    IOVector bias_gradient;

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

    void EnsureTempSizes();
};

Vs::GradVector CalculateNumericalGradient(const std::shared_ptr<Vs::Network> network, const Vs::IOVector &input,
                                          const Vs::IOVector &expected_output,
                                          const std::shared_ptr<Vs::ObjectiveFunction> objective_fn,
                                          const double param_step_size);
}  // namespace Vs

#endif
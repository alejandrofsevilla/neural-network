#pragma once

#include <cstddef>
#include <memory>
#include <vector>

class CostFunction;
class ActivationFunction;
class Neuron;

namespace options {
enum class ActivationFunctionType;
} // namespace options

class Layer {
public:
  Layer(std::size_t id, std::size_t numberOfInputs, std::size_t numberOfNeurons,
        options::ActivationFunctionType activationFunction);

  std::size_t id() const;
  std::size_t numberOfInputs() const;

  const std::vector<double> &inputs() const;

  const std::vector<Neuron> &neurons() const;

  const ActivationFunction &activationFunction() const;

  double computeLoss(const std::vector<double> &targets,
                     const CostFunction &costFunction) const;

  std::vector<double> computeOutputs(const std::vector<double> &inputs);
  std::vector<double> computeErrors(const Layer &nextLayer,
                                    const std::vector<double> &nextLayerErrors);
  std::vector<double> computeErrors(const std::vector<double> &targets,
                                    const CostFunction &costFunction);

  void updateNeuronWeights(std::size_t neuronId, double learnRate);
  void updateNeuronWeights(std::size_t neuronId,
                           const std::vector<double> &gradients,
                           double learnRate);

private:
  const std::size_t m_id;
  const std::size_t m_numberOfInputs;
  std::unique_ptr<ActivationFunction> m_activationFunction;
  std::vector<double> m_inputs;
  std::vector<Neuron> m_neurons;
};

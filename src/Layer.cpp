#include "Layer.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Neuron.h"
#include "Options.h"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace {
auto generateNeurons(const Layer &layer, std::size_t numberOfNeurons) {
  std::vector<Neuron> neurons;
  std::generate_n(std::back_inserter(neurons), numberOfNeurons,
                  [&neurons, &layer]() {
                    return Neuron{layer, neurons.size()};
                  });
  return neurons;
}
} // namespace

Layer::Layer(std::size_t id, std::size_t numberOfInputs,
             std::size_t numberOfNeurons,
             options::ActivationFunctionType activationFunction)
    : m_id{id}, m_numberOfInputs{numberOfInputs},
      m_activationFunction{ActivationFunction::instance(activationFunction)},
      m_inputs(numberOfInputs), m_neurons{
                                    generateNeurons(*this, numberOfNeurons)} {}

std::size_t Layer::id() const { return m_id; }

std::size_t Layer::numberOfInputs() const { return m_numberOfInputs; }

const std::vector<double> &Layer::inputs() const { return m_inputs; }

const std::vector<Neuron> &Layer::neurons() const { return m_neurons; }

const ActivationFunction &Layer::activationFunction() const {
  return *m_activationFunction;
}

double Layer::computeLoss(const std::vector<double> &targets,
                          const CostFunction &costFunction) const {
  return std::accumulate(m_neurons.cbegin(), m_neurons.cend(), 0.0,
                         [&targets, &costFunction](auto val, auto &n) {
                           return val + n.computeLoss(targets.at(n.id()),
                                                      costFunction);
                         }) /
         m_neurons.size();
}

std::vector<double> Layer::computeOutputs(const std::vector<double> &inputs) {
  m_inputs = inputs;
  std::vector<double> outputs;
  std::transform(m_neurons.begin(), m_neurons.end(),
                 std::back_inserter(outputs),
                 [](auto &n) { return n.computeOutput(); });
  return outputs;
}

std::vector<double>
Layer::computeErrors(const Layer &nextLayer,
                     const std::vector<double> &nextLayerErrors) {
  std::vector<double> errors;
  std::transform(m_neurons.begin(), m_neurons.end(), std::back_inserter(errors),
                 [&nextLayer, &nextLayerErrors](auto &n) {
                   return n.computeError(nextLayer, nextLayerErrors);
                 });
  return errors;
}

std::vector<double> Layer::computeErrors(const std::vector<double> &targets,
                                         const CostFunction &costFunction) {
  std::vector<double> errors;
  std::transform(m_neurons.begin(), m_neurons.end(), std::back_inserter(errors),
                 [&targets, &costFunction](auto &n) {
                   return n.computeError(targets.at(n.id()), costFunction);
                 });
  return errors;
}

void Layer::updateNeuronWeights(std::size_t neuronId, double learnRate) {
  m_neurons.at(neuronId).updateWeights(learnRate);
}

void Layer::updateNeuronWeights(std::size_t neuronId,
                                const std::vector<double> &gradients,
                                double learnRate) {
  m_neurons.at(neuronId).updateWeights(gradients, learnRate);
}

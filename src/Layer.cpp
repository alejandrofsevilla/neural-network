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
      m_inputs(numberOfInputs),
      m_errors(numberOfNeurons), m_neurons{
                                     generateNeurons(*this, numberOfNeurons)} {}

std::size_t Layer::id() const { return m_id; }

std::size_t Layer::numberOfInputs() const { return m_numberOfInputs; }

const std::vector<double> &Layer::inputs() const { return m_inputs; }

void Layer::setInputs(const std::vector<double> &inputs) { m_inputs = inputs; }

const std::vector<double> &Layer::errors() const { return m_errors; }

void Layer::setErrors(const std::vector<double> &errors) { m_errors = errors; }

const std::vector<Neuron> &Layer::neurons() const { return m_neurons; }

const ActivationFunction &Layer::activationFunction() const {
  return *m_activationFunction;
}

double Layer::computeLoss() const {
  return std::accumulate(m_neurons.cbegin(), m_neurons.cend(), 0.0,
                         [](auto val, auto &n) { return val + n.loss(); }) /
         m_neurons.size();
}

std::vector<double> Layer::computeOutputs() {
  std::vector<double> outputs;
  std::transform(m_neurons.begin(), m_neurons.end(),
                 std::back_inserter(outputs),
                 [](auto &n) { return n.computeOutput(); });
  return outputs;
}

std::vector<double> Layer::computeErrors(const Layer &nextLayer) {
  std::vector<double> errors;
  std::transform(m_neurons.begin(), m_neurons.end(), std::back_inserter(errors),
                 [&nextLayer](auto &n) { return n.computeError(nextLayer); });
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

void Layer::forwardPropagate(Layer &nextLayer) {
  nextLayer.setInputs(computeOutputs());
}

void Layer::backwardPropagate(Layer &prevLayer) {
  prevLayer.setErrors(prevLayer.computeErrors(*this));
}

void Layer::updateNeuronWeights(std::size_t neuronId, double learnRate) {
  m_neurons.at(neuronId).updateWeights(learnRate);
}

void Layer::updateNeuronWeights(std::size_t neuronId,
                                const std::vector<double> &gradients,
                                double learnRate) {
  m_neurons.at(neuronId).updateWeights(gradients, learnRate);
}

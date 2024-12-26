#include "Neuron.h"

#include <cassert>
#include <numeric>

Neuron::Neuron(std::size_t numberOfInputs,
               const ActivationFunction *activationFunction)
    : m_weights(numberOfInputs, 0.0), m_activationFunction(activationFunction) {
}

double &Neuron::weight(std::size_t pos) {
  assert(pos < m_weights.size());
  return m_weights.at(pos);
}

double Neuron::output(const std::vector<double> &inputs) const {
  assert(inputs.size() == m_weights.size());
  return m_activationFunction->operator()(intermediateQuantity(inputs));
}

double Neuron::outputDerivative(const std::vector<double> &inputs) const {
  assert(inputs.size() == m_weights.size());
  return m_activationFunction->derivative(intermediateQuantity(inputs));
}

double Neuron::intermediateQuantity(const std::vector<double> &inputs) const {
  return std::inner_product(inputs.cbegin(), inputs.cend(), m_weights.cbegin(),
                            0.0);
}

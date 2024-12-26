#include "Layer.h"

#include <cassert>

Layer::Layer(std::size_t numberOfInputs, std::size_t numberOfNeurons,
             ActivationFunction::Type activationFunctionType)
    : m_activationFunction{ActivationFunction::instance(
          activationFunctionType)},
      m_neurons(numberOfNeurons, {numberOfInputs, m_activationFunction.get()}) {
}

Neuron &Layer::neuron(std::size_t pos) {
  assert(pos < m_neurons.size());
  return m_neurons.at(pos);
}

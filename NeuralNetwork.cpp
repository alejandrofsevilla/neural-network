#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::size_t numberOfInputs)
    : m_numberOfOutputs{numberOfInputs}, m_layers{} {}

void NeuralNetwork::addLayer(std::size_t numberOfNeurons,
                             ActivationFunction::Type function) {
  m_layers.emplace_back(m_numberOfOutputs, numberOfNeurons, function);
  m_numberOfOutputs = numberOfNeurons;
}

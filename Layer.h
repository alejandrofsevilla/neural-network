#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "ActivationFunction.h"
#include "Neuron.h"

class Layer {
public:
  Layer(std::size_t numberOfInputs, std::size_t numberOfNeurons,
        ActivationFunction::Type activationFunctionType);

  Neuron &neuron(std::size_t pos);

private:
  std::unique_ptr<ActivationFunction> m_activationFunction;
  std::vector<Neuron> m_neurons;
};

#endif

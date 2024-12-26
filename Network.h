#ifndef NEURAL_NETWORK_NETWORK_H
#define NEURAL_NETWORK_NETWORK_H

#include "ActivationFunction.h"
#include "Layer.h"

class TrainingConfig;
class TrainingBatch;

class Network {
public:
  Network(std::size_t numberOfInputs);

  std::vector<double> outputs(const std::vector<double> &inputs);

  void addLayer(std::size_t numberOfNeurons,
                ActivationFunction::Type activationFunctionType);
  void train(TrainingConfig, const TrainingBatch &batch);

private:
  std::vector<Layer> m_layers;
};

#endif

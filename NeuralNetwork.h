#pragma once

#include "ActivationFunction.h"
#include "Layer.h"

class TrainingBatch;
class TrainingConfig;

class NeuralNetwork {
public:
  NeuralNetwork(std::size_t numberOfInputs);

  std::vector<double> outputs(const std::vector<double> &inputs) const;

  void addLayer(std::size_t numberOfNeurons, ActivationFunction::Type function);
  void train(TrainingConfig config, const TrainingBatch &batch);

private:
  std::size_t m_numberOfOutputs;
  std::vector<Layer> m_layers;
};

#pragma once

#include "Options.h"
#include "TrainingReport.h"
#include "TrainingSample.h"

#include <memory>
#include <utility>
#include <vector>

class Layer;

class NeuralNetwork {
public:
  explicit NeuralNetwork(std::size_t numberOfInputs);

  ~NeuralNetwork();

  std::vector<double> computeOutputs(const std::vector<double> &inputs);

  void addLayer(options::LayerConfig config);

  TrainingReport train(options::TrainingConfig config,
                       const TrainingBatch &batch);

private:
  const std::size_t m_numberOfInputs;
  std::size_t m_numberOfOutputs;
  std::vector<std::unique_ptr<Layer>> m_layers;
};

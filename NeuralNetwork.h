#pragma once

#include <memory>
#include <utility>
#include <vector>

class Layer;
struct TrainingBatch;
struct TrainingReport;

namespace options {
struct Layer;
struct Training;
} // namespace options

class NeuralNetwork {
public:
  NeuralNetwork();
  ~NeuralNetwork();

  std::vector<double> computeOutputs(const std::vector<double> &inputs);

  void addLayer(options::Layer opt);

  TrainingReport train(options::Training opt, const TrainingBatch &batch);

private:
  std::vector<Layer> m_layers;
};

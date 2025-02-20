#pragma once

#include <memory>
#include <utility>
#include <vector>

class Layer;
struct TrainingBatch;
struct TrainingReport;
namespace options {
struct LayerConfig;
struct TrainingConfig;
} // namespace options

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
  std::vector<Layer> m_layers;
};

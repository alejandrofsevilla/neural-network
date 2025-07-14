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
  NeuralNetwork() noexcept;
  ~NeuralNetwork() noexcept;

  void addLayer(options::Layer opt) noexcept;

  [[nodiscard]] std::vector<double>
  computeOutputs(const std::vector<double> &inputs) noexcept;

  TrainingReport train(options::Training opt,
                       const TrainingBatch &batch) noexcept;

private:
  std::vector<Layer> m_layers;
};

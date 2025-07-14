#pragma once

#include <cstddef>
#include <memory>
#include <vector>

class Layer;
class CostFunction;
struct TrainingReport;
struct TrainingBatch;

namespace options {
struct Training;
enum class Optimization;
enum class CostFunction;
} // namespace options

class OptimizationAlgorithm {
public:
  static std::unique_ptr<OptimizationAlgorithm>
  instance(options::Optimization optimization,
           options::CostFunction costFunction,
           std::vector<Layer> &layers) noexcept;

  virtual ~OptimizationAlgorithm() noexcept = default;

  TrainingReport run(TrainingBatch batch, std::size_t maxEpoch,
                     double learnRate, double lossGoal) noexcept;

protected:
  OptimizationAlgorithm(options::CostFunction costFunction,
                        std::vector<Layer> &layers) noexcept;

  virtual void afterSample() noexcept;
  virtual void afterEpoch() noexcept;

  void forwardPropagate(const std::vector<double> &inputs) noexcept;
  void backwardPropagate(const std::vector<double> &outputs) noexcept;
  void preprocess(TrainingBatch &batch) const noexcept;

  std::vector<Layer> &m_layers;
  std::unique_ptr<CostFunction> m_costFunction;
  std::size_t m_epochCount;
  std::size_t m_sampleCount;
  double m_learnRate;
  double m_loss;
};

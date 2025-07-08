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
           options::CostFunction costFunction, std::vector<Layer> &layers);

  virtual ~OptimizationAlgorithm() = default;

  TrainingReport run(TrainingBatch batch, std::size_t maxEpoch,
                     double learnRate, double lossGoal);

protected:
  OptimizationAlgorithm(options::CostFunction costFunction,
                        std::vector<Layer> &layers);

  virtual void afterSample();
  virtual void afterEpoch();

  void forwardPropagate(const std::vector<double> &inputs);
  void backwardPropagate(const std::vector<double> &outputs);
  void preprocess(TrainingBatch &batch) const;

  std::vector<Layer> &m_layers;
  std::unique_ptr<CostFunction> m_costFunction;
  std::size_t m_epochCount;
  std::size_t m_sampleCount;
  double m_learnRate;
  double m_loss;
};

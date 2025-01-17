#pragma once

#include <cstddef>
#include <memory>
#include <vector>

class CostFunction;
class Layer;
class Neuron;
class TrainingBatch;

namespace options {
class TrainingConfig;
enum class OptimizationType;
enum class CostFunctionType;
} // namespace options

class OptimizationAlgorithm {
public:
  static std::unique_ptr<OptimizationAlgorithm>
  instance(options::OptimizationType optimization,
           options::CostFunctionType costFunction,
           const std::vector<std::unique_ptr<Layer>> &layers);

  virtual ~OptimizationAlgorithm() = default;
  
  std::size_t epochsCount() const;

  double loss() const;

  virtual void run(TrainingBatch batch, std::size_t maxEpoch, double learnRate,
                   double lossGoal);

protected:
  virtual void beforeRun(double learnRate);
  virtual void afterSample();
  virtual void afterEpoch();

  OptimizationAlgorithm(options::CostFunctionType costFunction,
                        const std::vector<std::unique_ptr<Layer>> &layers);

  void updateLoss();
  void forwardPropagate(const std::vector<double> &inputs);
  void backwardPropagate(const std::vector<double> &outputs);
  void preprocess(TrainingBatch &batch) const;

  const std::vector<std::unique_ptr<Layer>> &m_layers;
  std::unique_ptr<CostFunction> m_costFunction;
  std::size_t m_epochsCount;
  std::size_t m_samplesCount;
  double m_learnRate;
  double m_loss;
};

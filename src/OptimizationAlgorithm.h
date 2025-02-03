#pragma once

#include <cstddef>
#include <memory>
#include <vector>

class CostFunction;
class Layer;
struct TrainingBatch;

namespace options {
struct TrainingConfig;
enum class OptimizationType;
enum class CostFunctionType;
} // namespace options

class OptimizationAlgorithm {
public:
  static std::unique_ptr<OptimizationAlgorithm>
  instance(options::OptimizationType optimization,
           options::CostFunctionType costFunction, std::vector<Layer> &layers);

  virtual ~OptimizationAlgorithm() = default;

  virtual void run(TrainingBatch batch, std::size_t maxEpoch, double learnRate,
                   double lossGoal);

  std::size_t epochsCount() const;

  double loss() const;

protected:
  OptimizationAlgorithm(options::CostFunctionType costFunction,
                        std::vector<Layer> &layers);

  virtual void beforeRun(double learnRate);
  virtual void afterSample();
  virtual void afterEpoch();

  void updateLoss();
  void forwardPropagate(const std::vector<double> &inputs);
  void backwardPropagate(const std::vector<double> &outputs);
  void preprocess(TrainingBatch &batch) const;

  std::vector<Layer> &m_layers;
  std::unique_ptr<CostFunction> m_costFunction;
  std::size_t m_epochsCount;
  std::size_t m_samplesCount;
  double m_learnRate;
  double m_loss;
};

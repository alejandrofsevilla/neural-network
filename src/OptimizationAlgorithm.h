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

  virtual void run(TrainingBatch batch, std::size_t maxEpoch, float learnRate,
                   float lossGoal);

  std::size_t epochsCount() const;

  float loss() const;

protected:
  OptimizationAlgorithm(options::CostFunctionType costFunction,
                        std::vector<Layer> &layers);

  virtual void beforeRun(float learnRate);
  virtual void afterSample();
  virtual void afterEpoch();

  void updateLoss();
  void forwardPropagate(const std::vector<float> &inputs);
  void backwardPropagate(const std::vector<float> &outputs);
  void preprocess(TrainingBatch &batch) const;

  std::vector<Layer> &m_layers;
  std::unique_ptr<CostFunction> m_costFunction;
  std::size_t m_epochsCount;
  std::size_t m_samplesCount;
  float m_learnRate;
  float m_loss;
};

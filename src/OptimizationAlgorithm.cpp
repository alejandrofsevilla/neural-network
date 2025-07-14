#include "OptimizationAlgorithm.h"

#include "ADAMOptimizationAlgorithm.h"
#include "CostFunction.h"
#include "GradientDescendOptimizationAlgorithm.h"
#include "Layer.h"
#include "Options.h"
#include "SGDOptimizationAlgorithm.h"
#include "TrainingReport.h"
#include "TrainingSample.h"

#include <Eigen/Dense>
#include <iostream>
#include <random>

OptimizationAlgorithm::OptimizationAlgorithm(
    options::CostFunction costFunction, std::vector<Layer> &layers) noexcept
    : m_layers{layers}, m_costFunction{CostFunction::instance(costFunction)},
      m_epochCount{0}, m_learnRate{0.0}, m_loss{0.0} {}

std::unique_ptr<OptimizationAlgorithm>
OptimizationAlgorithm::instance(options::Optimization type,
                                options::CostFunction costFunction,
                                std::vector<Layer> &layers) noexcept {
  switch (type) {
  case options::Optimization::ADAM:
    return std::make_unique<ADAMOptimizationAlgorithm>(costFunction, layers);
  case options::Optimization::SGD:
    return std::make_unique<SGDOptimizationAlgorithm>(costFunction, layers);
  case options::Optimization::GradientDescend:
  default:
    return std::make_unique<GradientDescendOptimizationAlgorithm>(costFunction,
                                                                  layers);
  }
}

TrainingReport OptimizationAlgorithm::run(TrainingBatch batch,
                                          std::size_t maxEpoch,
                                          double learnRate,
                                          double lossGoal) noexcept {
  auto begin{std::chrono::steady_clock::now()};
  preprocess(batch);
  m_learnRate = learnRate;
  auto &samples{batch.samples};
  for (m_epochCount = 0; m_epochCount < maxEpoch; m_epochCount++) {
    for (m_sampleCount = 0; m_sampleCount < samples.size(); m_sampleCount++) {
      auto &sample{samples.at(m_sampleCount)};
      forwardPropagate(sample.inputs);
      backwardPropagate(sample.outputs);
      afterSample();
    }
    if (m_loss < lossGoal) {
      break;
    }
    afterEpoch();
  }
  return {std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - begin),
          m_epochCount, m_loss};
}

void OptimizationAlgorithm::afterSample() noexcept {}

void OptimizationAlgorithm::afterEpoch() noexcept {}

void OptimizationAlgorithm::forwardPropagate(
    const std::vector<double> &inputs) noexcept {
  m_layers.front().updateOutputs(
      Eigen::Map<const Eigen::VectorXd>(inputs.data(), inputs.size()));
  std::for_each(m_layers.begin() + 1, m_layers.end(), [this](auto &l) {
    l.updateOutputs(m_layers.at(l.id() - 1).outputs());
  });
}

void OptimizationAlgorithm::backwardPropagate(
    const std::vector<double> &outputs) noexcept {
  auto &lastLayer{m_layers.back()};
  lastLayer.updateErrorsAndLoss(
      Eigen::Map<const Eigen::VectorXd>(outputs.data(), outputs.size()),
      *m_costFunction);
  m_loss += (lastLayer.loss() - m_loss) / (m_sampleCount + 1);
  std::for_each(m_layers.rbegin() + 1, m_layers.rend(),
                [this](auto &l) { l.updateErrors(m_layers.at(l.id() + 1)); });
}

void OptimizationAlgorithm::preprocess(TrainingBatch &batch) const noexcept {
  auto inputSize{m_layers.front().numberOfInputs()};
  auto outputSize{m_layers.back().numberOfNeurons()};
  for (auto it = batch.samples.begin(); it != batch.samples.end(); it++) {
    if (it->inputs.size() != inputSize || it->outputs.size() != outputSize) {
      std::cerr << "error: sample has incorrect dimensions." << std::endl;
      batch.samples.erase(it--);
    }
  }
  std::shuffle(batch.samples.begin(), batch.samples.end(),
               std::default_random_engine{});
}

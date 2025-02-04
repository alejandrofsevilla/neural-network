#include "OptimizationAlgorithm.h"

#include "ADAMOptimizationAlgorithm.h"
#include "CostFunction.h"
#include "GradientDescendOptimizationAlgorithm.h"
#include "Layer.h"
#include "Options.h"
#include "SGDOptimizationAlgorithm.h"
#include "TrainingSample.h"

#include <Eigen/Dense>
#include <iostream>
#include <random>

OptimizationAlgorithm::OptimizationAlgorithm(
    options::CostFunctionType costFunction, std::vector<Layer> &layers)
    : m_layers{layers}, m_costFunction{CostFunction::instance(costFunction)},
      m_epochsCount{0}, m_learnRate{0.0}, m_loss{0.0} {}

std::unique_ptr<OptimizationAlgorithm>
OptimizationAlgorithm::instance(options::OptimizationType type,
                                options::CostFunctionType costFunction,
                                std::vector<Layer> &layers) {
  switch (type) {
  case options::OptimizationType::ADAM:
    return std::make_unique<ADAMOptimizationAlgorithm>(costFunction, layers);
  case options::OptimizationType::SGD:
    return std::make_unique<SGDOptimizationAlgorithm>(costFunction, layers);
  case options::OptimizationType::GradientDescend:
  default:
    return std::make_unique<GradientDescendOptimizationAlgorithm>(costFunction,
                                                                  layers);
  }
}

void OptimizationAlgorithm::run(TrainingBatch batch, std::size_t maxEpoch,
                                float learnRate, float lossGoal) {
  beforeRun(learnRate);
  preprocess(batch);
  for (m_epochsCount = 0; m_epochsCount < maxEpoch; m_epochsCount++) {
    m_samplesCount = 0;
    std::for_each(batch.samples.begin(), batch.samples.end(), [this](auto &s) {
      m_samplesCount++;
      forwardPropagate(s.inputs);
      backwardPropagate(s.outputs);
      updateLoss();
      afterSample();
    });
    if (m_loss < lossGoal) {
      return;
    }
    afterEpoch();
  }
}

std::size_t OptimizationAlgorithm::epochsCount() const { return m_epochsCount; }

float OptimizationAlgorithm::loss() const { return m_loss; }

void OptimizationAlgorithm::beforeRun(float learnRate) {
  m_learnRate = learnRate;
  m_epochsCount = 0;
  m_loss = 0;
}

void OptimizationAlgorithm::afterSample() {}

void OptimizationAlgorithm::afterEpoch() {}

void OptimizationAlgorithm::updateLoss() {
  m_loss = m_loss + (m_layers.back().loss() - m_loss) / m_samplesCount;
}

void OptimizationAlgorithm::forwardPropagate(const std::vector<float> &inputs) {
  auto inputsView{
      Eigen::Map<const Eigen::VectorXf>(inputs.data(), inputs.size())};
  m_layers.front().updateOutputs(inputsView);
  std::for_each(m_layers.begin() + 1, m_layers.end(), [this](auto &l) {
    l.updateOutputs(m_layers.at(l.id() - 1).outputs());
  });
}

void OptimizationAlgorithm::backwardPropagate(
    const std::vector<float> &outputs) {
  auto &lastLayer{m_layers.back()};
  auto outputsView{
      Eigen::Map<const Eigen::VectorXf>(outputs.data(), outputs.size())};
  lastLayer.updateErrorsAndLoss(outputsView, *m_costFunction);
  std::for_each(m_layers.rbegin() + 1, m_layers.rend(),
                [this](auto &l) { l.updateErrors(m_layers.at(l.id() + 1)); });
}

void OptimizationAlgorithm::preprocess(TrainingBatch &batch) const {
  auto networkInputDimensions{m_layers.front().numberOfInputs()};
  auto networkOutputDimensions{m_layers.back().numberOfNeurons()};
  for (auto it = batch.samples.begin(); it != batch.samples.end(); it++) {
    if (it->inputs.size() != networkInputDimensions ||
        it->outputs.size() != networkOutputDimensions) {
      std::cerr << "error: sample has incorrect dimensions." << std::endl;
      batch.samples.erase(it--);
    }
  }
  std::shuffle(batch.samples.begin(), batch.samples.end(),
               std::default_random_engine{});
}

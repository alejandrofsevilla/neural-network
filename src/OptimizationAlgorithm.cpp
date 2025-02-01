#include "OptimizationAlgorithm.h"

#include "ADAMOptimizationAlgorithm.h"
#include "CostFunction.h"
#include "GradientDescendOptimizationAlgorithm.h"
#include "Layer.h"
#include "Neuron.h"
#include "Options.h"
#include "SGDOptimizationAlgorithm.h"
#include "TrainingSample.h"

#include <algorithm>
#include <iostream>
#include <random>

std::unique_ptr<OptimizationAlgorithm> OptimizationAlgorithm::instance(
    options::OptimizationType type, options::CostFunctionType costFunction,
    const std::vector<std::unique_ptr<Layer>> &layers) {
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

OptimizationAlgorithm::OptimizationAlgorithm(
    options::CostFunctionType costFunction,
    const std::vector<std::unique_ptr<Layer>> &layers)
    : m_layers{layers}, m_costFunction{CostFunction::instance(costFunction)},
      m_epochsCount{0}, m_learnRate{0.0}, m_loss{0.0} {}

double OptimizationAlgorithm::loss() const { return m_loss; }

void OptimizationAlgorithm::run(TrainingBatch batch, std::size_t maxEpoch,
                                double learnRate, double lossGoal) {
  beforeRun(learnRate);
  preprocess(batch);
  for (m_epochsCount = 0; m_epochsCount < maxEpoch; m_epochsCount++) {
    m_samplesCount = 0;
    std::for_each(batch.samples.cbegin(), batch.samples.cend(),
                  [this](auto &s) {
                    m_samplesCount++;
                    forwardPropagate(s.inputs);
                    backwardPropagate(s.outputs);
                    updateLoss(s.outputs);
                    afterSample();
                  });
    if (m_loss < lossGoal) {
      return;
    }
    afterEpoch();
  }
}

void OptimizationAlgorithm::beforeRun(double learnRate) {
  m_learnRate = learnRate;
  m_epochsCount = 0;
  m_loss = 0;
}

void OptimizationAlgorithm::afterSample() {}

void OptimizationAlgorithm::afterEpoch() {}

std::size_t OptimizationAlgorithm::epochsCount() const { return m_epochsCount; }

void OptimizationAlgorithm::updateLoss(const std::vector<double> &outputs) {
  auto loss = m_layers.back()->computeLoss(outputs, *m_costFunction);
  m_loss = m_loss + (loss - m_loss) / m_samplesCount;
}

void OptimizationAlgorithm::forwardPropagate(
    const std::vector<double> &inputs) {
  auto outputs{m_layers.front()->computeOutputs(inputs)};
  std::for_each(m_layers.begin() + 1, m_layers.end(),
                [&outputs](auto &l) { outputs = l->computeOutputs(outputs); });
}

void OptimizationAlgorithm::backwardPropagate(
    const std::vector<double> &outputs) {
  auto errors{m_layers.back()->computeErrors(outputs, *m_costFunction)};
  std::for_each(m_layers.rbegin() + 1, m_layers.rend(),
                [this, &errors](auto &l) {
                  errors = l->computeErrors(*m_layers.at(l->id() + 1), errors);
                });
}

void OptimizationAlgorithm::preprocess(TrainingBatch &batch) const {
  auto networkInputDimensions{m_layers.front()->numberOfInputs()};
  auto networkOutputDimensions{m_layers.back()->neurons().size()};
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

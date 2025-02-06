#include "NeuralNetwork.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Layer.h"
#include "OptimizationAlgorithm.h"
#include "Options.h"
#include "TrainingReport.h"
#include "TrainingSample.h"

#include <iostream>

NeuralNetwork::NeuralNetwork(std::size_t numberOfInputs)
    : m_numberOfInputs{numberOfInputs},
      m_numberOfOutputs(numberOfInputs), m_layers{} {}

NeuralNetwork::~NeuralNetwork() {}

std::vector<double>
NeuralNetwork::computeOutputs(const std::vector<double> &inputs) {
  if (inputs.size() != m_numberOfInputs) {
    std::cerr << "error: input vector has incorrect dimensions." << std::endl;
    return inputs;
  }
  m_layers.front().updateOutputs(
      Eigen::Map<const Eigen::VectorXd>(inputs.data(), inputs.size()));
  std::for_each(m_layers.begin() + 1, m_layers.end(), [this](auto &l) {
    l.updateOutputs(m_layers.at(l.id() - 1).outputs());
  });
  auto &outputs{m_layers.back().outputs()};
  return std::vector<double>(outputs.cbegin(), outputs.cend());
}

TrainingReport NeuralNetwork::train(options::TrainingConfig config,
                                    const TrainingBatch &batch) {
  auto optimizator{OptimizationAlgorithm::instance(
      config.optimization, config.costFunction, m_layers)};
  return optimizator->run(batch, config.maxEpoch, config.learnRate,
                          config.lossGoal);
}

void NeuralNetwork::addLayer(options::LayerConfig config) {
  m_layers.emplace_back(m_layers.size(), m_numberOfOutputs,
                        config.numberOfNeurons, config.activationFunction);
  m_numberOfOutputs = config.numberOfNeurons;
}

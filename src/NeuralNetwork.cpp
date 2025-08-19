#include "NeuralNetwork.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Layer.h"
#include "OptimizationAlgorithm.h"
#include "Options.h"
#include "TrainingReport.h"
#include "TrainingSample.h"

#include <iostream>

NeuralNetwork::NeuralNetwork() : m_layers{} {}

NeuralNetwork::~NeuralNetwork() = default;

std::vector<double>
NeuralNetwork::computeOutputs(const std::vector<double> &inputs) {
  if (m_layers.empty()) {
    return inputs;
  }
  if (inputs.size() != m_layers.front().numberOfInputs()) {
    std::cerr << "error: input vector has incorrect dimensions." << std::endl;
    return inputs;
  }
  m_layers.front().updateOutputs(
      Eigen::Map<const Eigen::VectorXd>(inputs.data(), inputs.size()));
  std::for_each(m_layers.begin() + 1, m_layers.end(), [this](auto &l) {
    l.updateOutputs(m_layers.at(l.id() - 1).outputs());
  });
  auto &outputs{m_layers.back().outputs()};
  return {outputs.cbegin(), outputs.cend()};
}

TrainingReport NeuralNetwork::train(options::Training opt,
                                    const TrainingBatch &batch) {
  auto optimizator{OptimizationAlgorithm::instance(opt.optimization,
                                                   opt.costFunction, m_layers)};
  return optimizator->run(batch, opt.maxEpoch, opt.learnRate, opt.lossGoal);
}

void NeuralNetwork::addLayer(options::Layer opt) {
  if (!m_layers.empty() &&
      opt.numberOfInputs != m_layers.back().numberOfNeurons()) {
    std::cerr << "error: layer has incorrect dimensions." << std::endl;
    return;
  }
  m_layers.emplace_back(m_layers.size(), opt.numberOfInputs,
                        opt.numberOfNeurons, opt.activationFunction);
}

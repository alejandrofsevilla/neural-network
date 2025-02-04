#include "NeuralNetwork.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Layer.h"
#include "OptimizationAlgorithm.h"
#include "Options.h"
#include "TrainingReport.h"
#include "TrainingSample.h"

#include <chrono>
#include <iostream>

NeuralNetwork::NeuralNetwork(std::size_t numberOfInputs)
    : m_numberOfInputs{numberOfInputs},
      m_numberOfOutputs(numberOfInputs), m_layers{} {}

NeuralNetwork::~NeuralNetwork() {}

std::vector<float>
NeuralNetwork::computeOutputs(const std::vector<float> &inputs) {
  if (inputs.size() != m_numberOfInputs) {
    std::cerr << "error: input vector has incorrect dimensions." << std::endl;
    return inputs;
  }
  auto outputs{Eigen::Map<const Eigen::VectorXf>(inputs.data(), inputs.size())};
  m_layers.front().updateOutputs(outputs);
  std::for_each(m_layers.begin() + 1, m_layers.end(), [this](auto &l) {
    l.updateOutputs(m_layers.at(l.id() - 1).outputs());
  });
  auto &lastLayerOutputs{m_layers.back().outputs()};
  return std::vector<float>(lastLayerOutputs.cbegin(), lastLayerOutputs.cend());
}

TrainingReport NeuralNetwork::train(options::TrainingConfig config,
                                    const TrainingBatch &batch) {
  auto begin = std::chrono::steady_clock::now();
  auto optimizator{OptimizationAlgorithm::instance(
      config.optimization, config.costFunction, m_layers)};
  optimizator->run(batch, config.maxEpoch, config.learnRate, config.lossGoal);
  auto end = std::chrono::steady_clock::now();
  return {std::chrono::duration_cast<std::chrono::milliseconds>(end - begin),
          optimizator->epochsCount(), optimizator->loss()};
}

void NeuralNetwork::addLayer(options::LayerConfig config) {
  m_layers.emplace_back(m_layers.size(), m_numberOfOutputs,
                        config.numberOfNeurons, config.activationFunction);
  m_numberOfOutputs = config.numberOfNeurons;
}

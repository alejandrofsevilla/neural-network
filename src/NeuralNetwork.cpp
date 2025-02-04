#include "NeuralNetwork.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Layer.h"
#include "Neuron.h"
#include "OptimizationAlgorithm.h"
#include "Options.h"
#include "TrainingSample.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>

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
  auto outputs{m_layers.front()->computeOutputs(inputs)};
  std::for_each(m_layers.begin() + 1, m_layers.end(),
                [&outputs](auto &l) { outputs = l->computeOutputs(outputs); });
  return outputs;
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
  m_layers.emplace_back(std::make_unique<Layer>(
      m_layers.size(), m_numberOfOutputs, config.numberOfNeurons,
      config.activationFunction));
  m_numberOfOutputs = config.numberOfNeurons;
}

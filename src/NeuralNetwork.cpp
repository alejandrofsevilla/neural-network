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
  m_layers.front()->setInputs(inputs);
  std::for_each(m_layers.begin(), m_layers.end() - 1, [this](auto &l) {
    l->forwardPropagate(m_layers.at(l->id() + 1).get());
  });
  return m_layers.back()->computeOutputs();
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

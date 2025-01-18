#pragma once

#include <cstddef>
#include <vector>

class ActivationFunction;
class CostFunction;
class Layer;

class Neuron {
public:
  Neuron(const Layer &layer, std::size_t id);

  std::size_t id() const;
  std::size_t layerId() const;

  const std::vector<double> &weights() const;
  const std::vector<double> &gradients() const;

  double loss() const;

  double computeOutput();
  double computeError(double target, const CostFunction &costFunction);
  double computeError(const Layer &nextLayer);

  void updateWeights(const std::vector<double> &gradients, double learnRate);
  void updateWeights(double learnRate);

private:
  void updateIntermediateQuantity();
  void updateGradients(double error);
  void updateLoss(double output, double target,
                  const CostFunction &costFunction);

  const ActivationFunction &m_activationFunction;
  const std::vector<double> &m_inputs;
  const std::size_t m_layerId;
  const std::size_t m_id;
  std::vector<double> m_weights;
  std::vector<double> m_gradients;
  double m_intermediateQty;
  double m_loss;
};

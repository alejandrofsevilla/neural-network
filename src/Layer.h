#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <memory>

class CostFunction;
class ActivationFunction;

namespace options {
enum class ActivationFunctionType;
} // namespace options

class Layer {
public:
  Layer(std::size_t id, std::size_t numberOfInputs, std::size_t numberOfNeurons,
        options::ActivationFunctionType activationFunction);

  std::size_t id() const;

  std::size_t numberOfInputs() const;

  std::size_t numberOfNeurons() const;

  double loss() const;

  const Eigen::VectorXd &outputs() const;

  const Eigen::VectorXd &errors() const;

  const Eigen::MatrixXd &weights() const;

  Eigen::MatrixXd computeGradients() const;

  void updateOutputs(const Eigen::VectorXd &inputs);

  void updateErrors(const Layer &nextLayer);

  void updateErrors(const Eigen::VectorXd &targets,
                    const CostFunction &costFunction);

  void updateWeights(double learnRate);

  void updateWeights(const Eigen::MatrixXd &gradients, double learnRate);

private:
  const std::size_t m_id;
  const std::size_t m_numberOfInputs;
  const std::size_t m_numberOfNeurons;
  std::unique_ptr<ActivationFunction> m_activationFunction;
  Eigen::VectorXd m_intermediateQtys;
  Eigen::VectorXd m_inputs;
  Eigen::VectorXd m_outputs;
  Eigen::VectorXd m_errors;
  Eigen::MatrixXd m_weights;
  double m_loss;
};

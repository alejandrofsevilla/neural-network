#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <memory>

class CostFunction;
class ActivationFunction;

namespace options {
enum class ActivationFunction;
} // namespace options

class Layer {
public:
  Layer(std::size_t id, std::size_t numberOfInputs, std::size_t numberOfNeurons,
        options::ActivationFunction activationFunction);

  std::size_t id() const;
  std::size_t numberOfInputs() const;
  std::size_t numberOfNeurons() const;

  double loss() const;

  const Eigen::MatrixXd &weights() const;
  const Eigen::VectorXd &inputs() const;
  const Eigen::VectorXd &outputs() const;
  const Eigen::VectorXd &errors() const;

  void updateWeights(double learnRate);
  void updateWeights(const Eigen::MatrixXd &gradients, double learnRate);
  void updateOutputs(const Eigen::VectorXd &inputs);
  void updateErrors(const Layer &nextLayer);
  void updateErrorsAndLoss(const Eigen::VectorXd &targets,
                           const CostFunction &costFunction);

private:
  const std::size_t m_id;
  const std::size_t m_numberOfInputs;
  const std::size_t m_numberOfNeurons;
  std::unique_ptr<ActivationFunction> m_activationFunction;
  Eigen::VectorXd m_outputDerivatives;
  Eigen::VectorXd m_outputs;
  Eigen::VectorXd m_inputs;
  Eigen::VectorXd m_errors;
  Eigen::MatrixXd m_weights;
  double m_loss;
};

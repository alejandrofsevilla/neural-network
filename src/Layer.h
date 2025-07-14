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
        options::ActivationFunction activationFunction) noexcept;

  std::size_t id() const noexcept;
  std::size_t numberOfInputs() const noexcept;
  std::size_t numberOfNeurons() const noexcept;

  double loss() const noexcept;

  const Eigen::MatrixXd &weights() const noexcept;
  const Eigen::VectorXd &inputs() const noexcept;
  const Eigen::VectorXd &outputs() const noexcept;
  const Eigen::VectorXd &errors() const noexcept;

  void updateWeights(double learnRate) noexcept;
  void updateWeights(const Eigen::MatrixXd &gradients,
                     double learnRate) noexcept;
  void updateOutputs(const Eigen::VectorXd &inputs) noexcept;
  void updateErrors(const Layer &nextLayer) noexcept;
  void updateErrorsAndLoss(const Eigen::VectorXd &targets,
                           const CostFunction &costFunction) noexcept;

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

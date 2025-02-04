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

  float loss() const;

  const Eigen::VectorXf &outputs() const;

  const Eigen::VectorXf &errors() const;

  const Eigen::MatrixXf &weights() const;

  Eigen::MatrixXf computeGradients() const;

  void updateOutputs(const Eigen::VectorXf &inputs);

  void updateErrors(const Layer &nextLayer);

  void updateErrors(const Eigen::VectorXf &targets,
                    const CostFunction &costFunction);

  void updateWeights(float learnRate);

  void updateWeights(const Eigen::MatrixXf &gradients, float learnRate);

private:
  const std::size_t m_id;
  const std::size_t m_numberOfInputs;
  const std::size_t m_numberOfNeurons;
  std::unique_ptr<ActivationFunction> m_activationFunction;
  Eigen::VectorXf m_intermediateQtys;
  Eigen::VectorXf m_inputs;
  Eigen::VectorXf m_outputs;
  Eigen::VectorXf m_errors;
  Eigen::MatrixXf m_weights;
  float m_loss;
};

#ifndef NEURAL_NETWORK_OPTIMIZATION_H
#define NEURAL_NETWORK_OPTIMIZATION_H

#include <memory>

class Optimization {
public:
  enum class Type { GradientDescend, ADAM };

  static std::unique_ptr<Optimization> instance(Type type);

  virtual double weightCorrection(const NeuronTrainingData &,
                                  const CostFunction *) = 0;
};

class GradientDescendOptimization : public Optimization {
  double
  weightCorrection(const NeuronTrainingData, &const CostFunction *) override;
};

class ADAMOptimization : public Optimization {
  double
  weightCorrection(const NeuronTrainingData, &const CostFunction *) override;
};

#endif

#ifndef NEURAL_NETWORK_OPTIMIZATION_H
#define NEURAL_NETWORK_OPTIMIZATION_H

#include <memory>

class Optimization {
public:
  enum class Type { GradientDescend, ADAM };

  static std::unique_ptr<Optimization> instance(Type type);

  virtual double correction(double gradient, std::size_t numberOfSamples) = 0;
};

class GradientDescendOptimization : public Optimization {
  double correction(double gradient, std::size_t numberOfSamples) override;
};

class ADAMOptimization : public Optimization {
  double correction(double gradient, std::size_t numberOfSamples) override;

private:
  double m_firstMomentEstimate{0.0};
  double m_secondMomentEstimate{0.0};
};

#endif

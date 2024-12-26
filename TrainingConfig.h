#ifndef NEURAL_NETWORK_TRAINING_CONFIG_H
#define NEURAL_NETWORK_TRAINING_CONFIG_H

#include "CostFunction.h"
#include "Optimization.h"

struct TrainingConfig {
  Optimization::Type optimization;
  CostFunction::Type costFunction;
  std::size_t maxEpoch;
  double learnRate;
  double lossGoal;
};

#endif

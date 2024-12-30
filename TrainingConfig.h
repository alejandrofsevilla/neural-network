#pragma once

#include "CostFunction.h"
#include "Optimization.h"

struct TrainingConfig {
  Optimization::Type optimization;
  CostFunction::Type costFunction;
  std::size_t maxEpoch;
  double learnRate;
  double lossGoal;
};

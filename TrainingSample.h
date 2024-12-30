#pragma once

#include <vector>

struct TrainingSample {
  std::vector<double> inputs;
  std::vector<double> outputs;
};

using TrainingBatch = std::vector<TrainingSample>;

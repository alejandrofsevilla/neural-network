#ifndef NEURAL_NETWORK_TRAINING_SAMPLE_H
#define NEURAL_NETWORK_TRAINING_SAMPLE_H

#include <vector>

using TrainingBatch = std::vector<TrainingSample>;

struct TrainingSample {
  std::vector<double> inputs;
  std::vector<double> outputs;
};

#endif

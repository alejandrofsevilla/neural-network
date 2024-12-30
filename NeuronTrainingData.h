#ifndef NEURAL_NETWORK_NEURON_TRAINING_DATA_H
#define NEURAL_NETWORK_NEURON_TRAINING_DATA_H

#include <vector>

struct NeuronTrainingData {
  double output;
  double outputDerivative;
  double target;
};

using NeuronTrainingDataBatch = std::vector<NeuronTrainingData>;

#endif

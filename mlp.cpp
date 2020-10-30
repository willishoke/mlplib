// Willis Hoke
// Multi-layer Perceptron Algorithm for MNIST
// With stochastic gradient descent

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <ctime>

// Seed the global RNG with current time
std::default_random_engine gen(time(0));

// Initialize probability distribution for weights
std::uniform_real_distribution<double> dist(-0.05, 0.05);

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// Hold a single image's pixel data and label
struct Image 
{
  Vector data;
  uint8_t label;
};

using Dataset = std::vector<Image>;

// Hold parameter values for training
struct Params
{
  Params(double e, double a, double ep) :
    eta(e), alpha(a), epochs(ep)
    { } 

  double eta;
  double alpha;
  int epochs;
};

// Stores vectors containing weights, biases, activations, etc
struct Model
{
  Matrix hiddenWeights;     // n x 784
  Matrix outputWeights;     // 10 x n
  Vector hiddenBiases;      // n 
  Vector outputBiases;      // 10
  Vector hiddenActivations; // n
  Vector outputActivations; // 10
  Vector hiddenErrors;      // n
  Vector outputErrors;      // 10
  Matrix hiddenDeltas;      // n x 784
  Matrix outputDeltas;      // 10 x n
  Vector hiddenBiasDeltas;  // n
  Vector outputBiasDeltas;  // 10

  Model(int n)
  {
    // Init to n x 784
    // Fill with values in range (-0.05, 0.05)
    hiddenWeights.resize(n);
    for (auto& ws : hiddenWeights) {
      ws.resize(784);
      for (auto& w : ws) {
        w = dist(gen);
      }
    } 
    // Init to 10 x n
    // Fill with values in range (-0.05, 0.05)
    outputWeights.resize(10);
    for (auto& ws : outputWeights) {
      ws.resize(n);
      for (auto& w : ws) {
        w = dist(gen);
      }
    }

    hiddenBiases.resize(n);
    for (auto& b : hiddenBiases) {
      b = dist(gen);
    }
    outputBiases.resize(10); 
    for (auto& b : outputBiases) {
      b = dist(gen);
    }

    hiddenActivations.resize(n, 0);
    outputActivations.resize(10, 0);

    hiddenErrors.resize(n, 0.0);
    outputErrors.resize(10, 0.0);

    // Init to size n x 784, fill with zeroes
    hiddenDeltas.resize(n);
    for (auto& ds : hiddenDeltas) {
      ds.resize(784, 0.0);
    } 
    // Init to size 10 x n, fill with zeroes
    outputDeltas.resize(10);
    for (auto& ds : outputDeltas) {
      ds.resize(n, 0.0);
    } 

    // Init to size n, fill with zeroes
    hiddenBiasDeltas.resize(n, 0.0);
    // Init to size 10, fill with zeroes
    outputBiasDeltas.resize(10, 0.0);
  }
};


// Calculate the value of the sigmoid activation function
double sigmoid
( const double x )
{
  return 1.0 / (1.0 + exp(-x));
}

// Read a file containing labelled data in CSV format
Dataset readFile
( const std::string filepath )
{
  Dataset dataset; 
  std::string line, label, value;
  // Open a filestream for parsing
  std::ifstream in(filepath);
  // Dump column headers
  std::getline(in, line);
  // Iterate through CSV line by line
  while (std::getline(in, line)) {
    Image image;
    // Create a stream for each line
    std::stringstream ss(line);
    // Get first value (label)
    std::getline(ss, label, ',');
    // Parse as integer, push to label vector
    image.label = std::stoi(label);
    // Parse remaining values as doubles
    while (ss.good()) {
      getline(ss, value, ',');
      image.data.push_back(std::stod(value) / 255.0);
    }
    // Push image vector to data vector
    dataset.push_back(image);

  }
  return dataset;
}

// Get the index of the maximum value in a vector
int maxIndex
( const Vector & v ) 
{
  return std::distance
  ( v.begin()
  , std::max_element(v.begin(), v.end())
  ) ;
}

// Propogate data through an entire layer and update activations
void forwardProp 
( const Vector & data
, const Matrix & weights
, const Vector & biases
, Vector & activations
)
{
  for (auto i = 0; i < weights.size(); ++i) {
    double sum = biases.at(i);
    for (auto n = 0; n < data.size(); ++n) {
      sum += weights.at(i).at(n) * data.at(n);
    }
    activations.at(i) = sigmoid(sum);
  }
}

// Train on a single training example
void train 
( const Image & image, Model & model, const Params & params)
{
  // Forward propogation step
  forwardProp
    (image.data, model.hiddenWeights, model.hiddenBiases, model.hiddenActivations); 
  forwardProp
    (model.hiddenActivations, model.outputWeights, model.outputBiases, model.outputActivations);
 
  // Calculate output errors 
  for (auto k = 0; k < model.outputErrors.size(); ++k) {
    double output = model.outputActivations.at(k);
    double target = k == image.label ? 0.9 : 0.1;
    model.outputErrors.at(k) = output * (1.0 - output) * (target - output);
  }

  // Calculate hidden errors
  for (auto j = 0; j < model.hiddenErrors.size(); ++j) {
    double output = model.hiddenActivations.at(j);
    double sum = 0;
    for (auto k = 0; k < model.outputErrors.size(); ++k) {
      sum += model.hiddenWeights.at(k).at(j) * model.outputErrors.at(k); 
    }
    model.hiddenErrors.at(j) = output * (1.0 - output) * sum;
  }

  // Calculate hidden gradients and update weights
  for (auto k = 0; k < model.outputErrors.size(); ++k) {
    for (auto j = 0; j < model.hiddenActivations.size(); ++j) {
      model.outputDeltas.at(k).at(j) = 
        (params.eta * model.outputErrors.at(k) * model.hiddenActivations.at(j)) +
        (params.alpha * model.outputDeltas.at(k).at(j));
      model.outputWeights.at(k).at(j) += model.outputDeltas.at(k).at(j);
    }
    model.outputBiasDeltas.at(k) = 
      (params.eta * model.outputErrors.at(k)) +
      (params.alpha * model.outputBiasDeltas.at(k));
    model.outputBiases.at(k) += model.outputBiasDeltas.at(k);
  }

  // Calculate output gradients and update weights
  for (auto j = 0; j < model.hiddenErrors.size(); ++j) {
    for (auto i = 0; i < image.data.size(); ++i) {
      model.hiddenDeltas.at(j).at(i) = 
        (params.eta * model.hiddenErrors.at(j) * image.data.at(i)) +
        (params.alpha * model.hiddenDeltas.at(j).at(i));
      model.hiddenWeights.at(j).at(i) += model.hiddenDeltas.at(j).at(i);
    }
    model.hiddenBiasDeltas.at(j) = 
      (params.eta * model.hiddenErrors.at(j)) + 
      (params.alpha * model.hiddenBiasDeltas.at(j));
    model.hiddenBiases.at(j) += model.hiddenBiasDeltas.at(j);
  }
}

// Train for a single epoch
void epoch
(Dataset & dataset, Model & model, const Params & params)
{
  // Shuffle dataset before training
  std::shuffle
    (dataset.begin(), dataset.end(), gen);
  for (auto & image : dataset) {
    train(image, model, params); 
  }
}

// Test accuracy of model on dataset
double testAccuracy
(const Dataset & dataset, Model & model)
{
  auto counter = 0;
  int examples = dataset.size();
 
  for (auto i = 0; i < examples; ++i) { 
    forwardProp
      (dataset.at(i).data, model.hiddenWeights, model.hiddenBiases, model.hiddenActivations);
    forwardProp
      (model.hiddenActivations, model.outputWeights, model.outputBiases, model.outputActivations);
    uint8_t classification = maxIndex(model.outputActivations);
    if (classification == dataset.at(i).label) ++counter;
  }
  return (double) counter / examples;
}

// Generate a confusion matrix
std::vector<std::vector<int>> confusionMatrix
( const Dataset & dataset, Model & model )
{
  // Generate 10 x 10 matrix filled with zeroes
  std::vector<std::vector<int>> m(10);
  for (auto & v : m) {
    v.resize(10, 0);
  }

  // Run model on entire dataset
  for (auto & image : dataset) {
    forwardProp
      (image.data, model.hiddenWeights, model.hiddenBiases, model.hiddenActivations);
    forwardProp
      (model.hiddenActivations, model.outputWeights, model.outputBiases, model.outputActivations);
    auto classification = maxIndex(model.outputActivations);
    m.at(classification).at(image.label) += 1;
  }
  return m;
}

// Output a matrix, with padding on columns
void printMatrix
(std::vector<std::vector<int>> & m)
{
  for (auto & row : m) {
    for (auto & col : row) {
      std::cout << col;
      for (auto i = 0; i < 5 - std::to_string(col).size(); ++i) {
        std::cout << ' ';
      }
    }
    std::cout << std::endl;
  }
}

// Print distribution of data in dataset
void dataStats
( Dataset & data ) 
{
  std::vector<int> counts(10, 0);
  for (auto & d : data) {
    counts.at(d.label) += 1;
  }
  for (auto & c : counts) {
    std::cout << c << ' ';
  }
  std::cout << std::endl;
}

// Reduce the number of examples in a dataset
void truncateDataset
( int size, Dataset & dataset )
{
  std::shuffle
    (dataset.begin(), dataset.end(), gen);
  dataset.resize(size); 
}

int main
( int argc, char** argv )
{
  int hiddenNeurons = 100;
  Model model(hiddenNeurons);

  // Learning rate
  double eta = 0.1;

  // Momentum
  double alpha = 0.9;

  // Number of training epochs
  int epochs = 50;

  // Initialize parameters
  Params params(eta, alpha, epochs);

  // Parse train and test datasets
  Dataset trainData = readFile("./data/mnist_train.csv");
  Dataset testData = readFile("./data/mnist_test.csv");

  // Use this for testing smaller datasets:
  // truncateDataset(15000, trainData);

  std::ofstream results("results.txt");

  // Test accuracy before training 
  double testAcc = testAccuracy(testData, model);
  double trainAcc = testAccuracy(trainData, model);
  
  results << testAcc << ' ' << trainAcc << std::endl;
  std::cout << testAcc << "\t\t" << trainAcc << std::endl;

  // Train network
  for (int i = 0; i < epochs; ++i) {
    epoch(trainData, model, params);
    testAcc = testAccuracy(testData, model);
    trainAcc = testAccuracy(trainData, model);
    results << testAcc << ' ' << trainAcc << std::endl;  
    std::cout << testAcc << "\t\t" << trainAcc << std::endl;
  }

  // Generate confusion matrix
  auto m = confusionMatrix(testData, model);
  printMatrix(m);
 
  results.close(); 

  return 0;
}

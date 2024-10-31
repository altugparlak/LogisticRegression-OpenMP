#ifndef TEST_H
#define TEST_H
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * @brief Tests the propagation function with predefined weight, bias, and data.
 * Initializes weights and biases, runs the propagation function, and outputs
 * the gradients (`dw`, `db`) and cost for validation.
 */
void propagation_test1();

/**
 * @brief Tests the propagation function using externally provided parameters.
 * Accepts custom weight, bias, and data sets to run the propagation function,
 * printing partial gradients and cost for validation with larger data.
 * 
 * @param w Vector of weights
 * @param b Vector of bias terms
 * @param train_set Input data set
 * @param true_label_set Labels for training data
 */
void propagation_test2(const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set);


/**
 * @brief Tests the optimize function with predefined parameters and data.
 * Initializes weights, biases, and dataset, then runs the optimization function,
 * outputting progress and results for gradient descent validation.
 */
void optimization_test1();

/**
 * @brief Tests the optimize function using externally provided parameters.
 * Accepts custom weight, bias, data sets, number of iterations, and learning rate
 * for optimization testing, allowing performance validation on larger datasets.
 * 
 * @param w Vector of weights
 * @param b Vector of bias terms
 * @param train_set Input data set
 * @param true_label_set Labels for training data
 * @param num_iterations Number of optimization iterations
 * @param learning_rate Step size for optimization
 */
void optimization_test2(const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate);

/**
 * @brief Tests the prediction function with predefined parameters and data.
 * Initializes weights, biases, and input data to run prediction, displaying
 * output predictions for model evaluation.
 */
void prediction_test1();

/**
 * @brief Tests the prediction functionality after optimization.
 * Runs optimization with specified parameters, extracts optimized weight
 * and bias values, then sets up prediction to evaluate model performance
 * on training data.
 * 
 * @param w Vector of weights
 * @param b Vector of bias terms
 * @param train_set Input data set
 * @param true_label_set Labels for training data
 * @param num_iterations Number of optimization iterations
 * @param learning_rate Step size for optimization
 */
void prediction_test2(const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate);

#endif

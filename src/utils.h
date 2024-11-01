#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <numeric>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace std::chrono;

/**
 * @brief Computes the sigmoid activation function for a given input.
 * @param x Input value.
 * @return Sigmoid activation of x.
 */
float activation(const float& x);

/**
 * @brief Computes a safe logarithm of the input by clipping values close to zero.
 * @param x Input value.
 * @return Logarithm of x with clipping to avoid log(0).
 */
float safe_log(float x);


/**
 * @brief Loads all .jpg images from the specified directory path.
 * @param images_path Path to the directory containing images.
 * @return A vector of loaded images as cv::Mat objects.
 */
vector<cv::Mat> getImages(const string& images_path);

/**
 * @brief Flattens and normalizes each image in the input vector.
 * @param images A vector of images to be flattened.
 * @return A vector of flattened and normalized images.
 */
vector<cv::Mat> getFlattenImages(const vector<cv::Mat>& images);

/**
 * @brief Computes the gradient and cost for logistic regression using forward and backward propagation.
 * @param w Weights vector.
 * @param b Bias vector.
 * @param train_set Training set of images.
 * @param true_label_set True labels for each training sample.
 * @return A pair containing gradients and computed cost.
 */
pair<unordered_map<string, vector<float>>, float> propagation(
    const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set);

/**
 * @brief Optimizes weights and biases for logistic regression by iterating gradient descent.
 * @param w Initial weights.
 * @param b Initial bias.
 * @param train_set Training set of images.
 * @param true_label_set True labels for each training sample.
 * @param num_iterations Number of iterations for optimization.
 * @param learning_rate Learning rate for gradient descent.
 * @return A tuple containing updated weights, gradients, and cost history.
 */
tuple<unordered_map<string, vector<float>>, 
    unordered_map<string, vector<float>>, vector<float>> optimize(
    const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate);

/**
 * @brief Predicts binary labels for a set of input images using learned weights and bias.
 * @param w Weights vector.
 * @param b Bias vector.
 * @param X_input Input images.
 * @return A vector of predicted labels (0 or 1).
 */
vector<int> predict(const vector<float>& w, const vector<float>& b, const vector<cv::Mat>& X_input);

/**
 * @brief Calculates accuracy by comparing predictions with true labels.
 * @param Y_prediction Predicted labels.
 * @param Y True labels.
 * @return Accuracy as a percentage.
 */
double calculate_accuracy(const vector<int>& Y_prediction, const vector<int>& Y);

#endif

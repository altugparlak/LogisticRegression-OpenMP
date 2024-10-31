#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;

float activation(const float& x);

vector<cv::Mat> getImages(const string& images_path);

vector<cv::Mat> getFlattenImages(const vector<cv::Mat>& images);

void normalize(vector<cv::Mat>& images);

pair<unordered_map<string, vector<float>>, float> propagation(
    const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set);

tuple<unordered_map<string, vector<float>>, 
    unordered_map<string, vector<float>>, vector<float>> optimize(
    const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate);

vector<int> predict(const vector<float>& w, const vector<float>& b, const vector<cv::Mat>& X_input);

#endif

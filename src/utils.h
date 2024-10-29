#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <omp.h>
#include <unordered_map>

using namespace std;

double activation(const double& x);

vector<cv::Mat> getImages(const string& images_path);

vector<cv::Mat> getFlattenImages(const vector<cv::Mat>& images);

int normalize(vector<cv::Mat>& images);

pair<unordered_map<string, vector<float>>, float> propagation(const vector<float>& w, const vector<float>& b, const vector<cv::Mat>& train_set, const vector<int>& true_label_set);

#endif

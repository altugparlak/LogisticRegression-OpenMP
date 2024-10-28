#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <omp.h>

using namespace std;

double activation(const double& x);

vector<cv::Mat> getImages(const string& images_path);

vector<cv::Mat> getFlattenImages(const vector<cv::Mat>& images);

int normalize(vector<cv::Mat>& images);

#endif

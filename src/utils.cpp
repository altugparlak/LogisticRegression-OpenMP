#include "utils.h"
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

double activation(const double& x) {
    return 1.0 / (1.0 + exp(-x));
}

vector<cv::Mat> getImages(const string& images_path) {
    vector<cv::Mat> images;

    for (const auto& entry : fs::directory_iterator(images_path)) {
        image_path = entry.path();
        
        if (image_path.extension() == ".jpg") {
            cv::Mat img = cv::imread(image_path.string());
            if (!img.empty()) {
                images.push_back(img);
            } else {
                cerr << "Failed to read image: " << entry.path().string() << endl;
            }
        }
    }

    return images;
}

vector<cv::Mat> getFlattenImages(const vector<cv::Mat>& images) {
    vector<cv::Mat> flatten_images;
    flatten_images.reserve(images.size());

    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++) {
        if (images[i].empty()) {
            cerr << "Warning: Image at index " << i << " is empty and will be skipped." << endl;
            continue;
        }
        cv::Mat flatten_image = images[i].reshape(1, 1);

        #pragma omp critical
        flatten_images.push_back(flatten_image);
    }

    return flatten_images;
}

int normalize(vector<cv::Mat>& images) {
    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++) {
        if (images[i].empty()) {
            cerr << "Warning: Image at index " << i << " is empty and will be skipped during normalization." << endl;
            continue;
        }
        
        images[i].convertTo(images[i], CV_32F, 1.0 / 255.0);
    }
    return 0;
}

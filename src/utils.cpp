#include "utils.h"
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

/**
 * @brief Sigmoid activation function.
 */
double activation(const double& x) {
    return 1.0 / (1.0 + exp(-x));
}

vector<cv::Mat> getImages(const string& images_path) {
    vector<cv::Mat> images;

    for (const auto& entry : fs::directory_iterator(images_path)) {
        auto image_path = entry.path();
        
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

pair<unordered_map<string, vector<float>>, float> propagation(
    const vector<float>& w, const vector<float>& b,
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set) {
    
    int m = train_set.size();
    float cost = 0.0;
    unordered_map<string, vector<float>> grads;
    
    vector<float> dw(w.size(), 0.0);
    vector<float> db_sum(b.size(), 0.0);

    // Forward propagation
    #pragma omp parallel for reduction(+:cost)
    for (int i = 0; i < m; i++) {
        float z = 0.0;
        cv::Mat X = train_set[i];
        
        for (int j = 0; j < w.size(); j++) {
            // w^T * x
            z += w[j] * X.at<float>(j);
        }
        
        z += b[i];

        // Sigmoid activation
        float A = activation(z);
        
        // Compute cost
        float y = true_label_set[i];
        cost += -y * log(A) - (1 - y) * log(1 - A);
        
        // Calculate gradients
        float dz = A - y;
        for (int j = 0; j < w.size(); j++) {
            // dz^t * X
            dw[j] += dz * X.at<float>(j);
            db_sum[j] += dz;
        }
    }

    cost /= m;
    for (int i = 0; i < w.size(); i++) {
        dw[i] /= m;
        db_sum[i] /= m;
    }

    grads["dw"] = dw;
    grads["db"] = db_sum;

    return make_pair(grads, cost);
}

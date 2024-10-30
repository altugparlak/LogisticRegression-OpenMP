#include "utils.h"
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

/**
 * @brief Sigmoid activation function.
 */
float activation(const float& x) {
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

void normalize(vector<cv::Mat>& images) {
    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++) {
        if (images[i].empty()) {
            cerr << "Warning: Image at index " << i << " is empty and will be skipped during normalization." << endl;
            continue;
        }
        
        images[i].convertTo(images[i], CV_32F, 1.0 / 255.0);
    }
    return;
}

pair<unordered_map<string, vector<float>>, float> propagation(
    const vector<float>& w, const vector<float>& b,
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set) {
    
    int m_train = train_set.size();
    cv::Size size = train_set[0].size();
    int m = size.width*size.height;
    float cost = 0.0;
    unordered_map<string, vector<float>> grads;
    
    vector<float> dw(m_train, 0.0);
    vector<float> db_sum(1, 0.0);

    vector<float> z(m, 0.0);
    vector<float> dz(m, 0.0);
    vector<float> A(m, 0.0);

    #pragma omp parallel
	{
		#pragma omp for
		for(int i = 0; i < m_train; i++)
		{
			cv::Mat X = train_set[i];

            for (int k = 0; k < m; k++)
            {
                z[k] += w[i] * X.at<float>(k);
            }
		}
        #pragma omp barrier

        #pragma omp for
        for (int k = 0; k < m; k++)
        {
            z[k] += b[0];
            A[k] = activation(z[k]);
        }
        
        #pragma omp barrier

        #pragma omp for
		for(int i = 0; i < m; i++)
		{
            float y = true_label_set[i];

            #pragma omp critical
            cost += y * log(A[i]) + (1 - y) * log(1 - A[i]);
		}
        #pragma omp barrier

        #pragma omp for
		for(int i = 0; i < m; i++)
		{
            float y = true_label_set[i];
            dz[i] = A[i] - y;
        }
        #pragma omp barrier

        #pragma omp for
        for (int j = 0; j < m_train; j++) {
            cv::Mat X = train_set[j];

            for (int k = 0; k < m; k++)
            {
                dw[j] += dz[k] * X.at<float>(k);
            }
            #pragma omp critical
            db_sum[0] += dz[j];
        }
        #pragma omp barrier

        #pragma omp single
        {
            cost /= -m;
        }

        #pragma omp for
        for (int i = 0; i < m_train; i++) {
            dw[i] /= m;
        }
        #pragma omp single 
        {
            db_sum[0] /= m;
        }

        #pragma omp barrier
	}

    grads["dw"] = dw;
    grads["db"] = db_sum;

    return make_pair(grads, cost);
}

tuple<unordered_map<string, vector<float>>, 
    unordered_map<string, vector<float>>, vector<float>> optimize(
    const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate) {
    
    unordered_map<string, vector<float>> grads;
    unordered_map<string, vector<float>> params;
    vector<float> w_copy = w;
    vector<float> b_copy = b;

    vector<float> costs(num_iterations, 0.0);
    vector<float> dw;
    vector<float> db;

    for (int i = 0; i < num_iterations; i++)
    {
        auto result = propagation(w_copy, b_copy, train_set, true_label_set);

        // Access results
        auto grads = result.first;
        dw = grads["dw"];
        db = grads["db"];
        float cost = result.second;

        #pragma omp parallel for
        for (int j = 0; j < w.size(); j++)
        {
            w_copy[j] = w_copy[j] - learning_rate * dw[j];
            b_copy[0] = b_copy[0] - learning_rate * db[0];
        }
        #pragma omp barrier
        cout << "Cost after iteration " << i << ": " << cost << endl;
        if ((i % 100) == 0) {
            costs.push_back(cost);
            cout << "Cost after iteration " << i << ": " << cost << endl;
        }
    }

    params["w"] = w_copy;
    params["b"] = b_copy;
    
    grads["dw"] = dw;
    grads["db"] = db;
    return make_tuple(params, grads, costs);
}
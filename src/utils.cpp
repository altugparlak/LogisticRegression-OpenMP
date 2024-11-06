#include "utils.h"
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

float activation(const float& x) {
    return 1.0 / (1.0 + exp(-x));
}

float safe_log(float x) {
    // Clip to avoid undefined behavior with log(0)
    return log(max(x, 1e-10f));
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

    for (int i = 0; i < images.size(); i++) {
        if (images[i].empty()) {
            cerr << "Warning: Image at index " << i << " is empty and will be skipped." << endl;
            continue;
        }
        cv::Mat flatten_image = images[i].reshape(1, 1);
        flatten_image.convertTo(flatten_image, CV_32F);
        flatten_image /= 255.0;
        flatten_images.push_back(flatten_image);
    }

    return flatten_images;
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
    vector<float> db(1, 0.0);

    vector<float> z(m, 0.0);
    vector<float> dz(m, 0.0);
    vector<float> A(m, 0.0);

    #pragma omp parallel shared(cost) shared(db)
	{
		#pragma omp for schedule(dynamic)
		for(int i = 0; i < m_train; i++) {
			cv::Mat X = train_set[i];
            for (int k = 0; k < m; k++) {
                z[k] += w[i] * X.at<float>(k);
            }
		}
        #pragma omp barrier

        #pragma omp for schedule(dynamic)
        for (int k = 0; k < m; k++) {
            z[k] += b[0];
            A[k] = activation(z[k]);
            // Clip A[k] to avoid exact 0 or 1
        }
        #pragma omp barrier

        float local_cost = 0.0;
        #pragma omp for schedule(dynamic)
        for(int i = 0; i < m; i++) {
            float y = true_label_set[i];
            local_cost += y * safe_log(A[i]) + (1 - y) * safe_log(1 - A[i]);
        }

        #pragma omp critical
        {
            cost += local_cost;
        }

        #pragma omp barrier

        #pragma omp for schedule(dynamic)
		for(int i = 0; i < m; i++) {
            float y = true_label_set[i];
            dz[i] = A[i] - y;
        }
        #pragma omp barrier

        #pragma omp for schedule(dynamic)
        for (int j = 0; j < m_train; j++) {
            cv::Mat X = train_set[j];

            for (int k = 0; k < m; k++) {
                dw[j] += dz[k] * X.at<float>(k);
            }
        }
        #pragma omp barrier
        
        #pragma omp single
        {
            for (int j = 0; j < m; j++) {
                db[0] += dz[j];
            }
        }
        #pragma omp barrier

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < m_train; i++) {
            dw[i] /= m;
        }

        #pragma omp single
        {
            cost /= -m;
            db[0] /= m;
        }

        #pragma omp barrier
	}

    grads["dw"] = dw;
    grads["db"] = db;

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

    vector<float> costs;
    vector<float> dw;
    vector<float> db;

    for (int i = 0; i < num_iterations; i++) {
        auto result = propagation(w_copy, b_copy, train_set, true_label_set);

        // Access results
        auto grads = result.first;
        dw = grads["dw"];
        db = grads["db"];
        float cost = result.second;

        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < w_copy.size(); j++)
        {
            w_copy[j] = w_copy[j] - learning_rate * dw[j];
            b_copy[0] = b_copy[0] - learning_rate * db[0];
        }

        #pragma omp barrier

        #pragma omp single
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

vector<int> predict(const vector<float>& w, const vector<float>& b, const vector<cv::Mat>& X_input) {

    cv::Size size = X_input[0].size();
    int m = size.width*size.height;
    int m_train = X_input.size();

    vector<int> Y_prediction(m, 0);
    vector<float> z(m, 0.0);
    vector<float> A(m, 0.0);

    #pragma omp parallel
	{
		#pragma omp for schedule(dynamic)
		for(int i = 0; i < m_train; i++) {
			cv::Mat X = X_input[i];
            
            // np.dot(w.T, X)
            for (int k = 0; k < m; k++) {
                z[k] += w[i] * X.at<float>(k);
            }
		}
        #pragma omp barrier

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < m; i++) {
            // z + b
            z[i] += b[0];
            // sigmoid(z) --- > A = sigmoid(np.dot(w.T, X) + b)
            A[i] = activation(z[i]);
            if (A[i] > 0.5) 
                Y_prediction[i] = 1;

        }
        #pragma omp barrier
    }

    return Y_prediction;
}

double calculate_accuracy(const std::vector<int>& Y_prediction, const std::vector<int>& Y) {
    if (Y_prediction.size() != Y.size()) {
        std::cerr << "Error: Vectors must be the same size." << std::endl;
        return -1.0;
    }

    double mean_abs_error = 0.0;
    for (size_t i = 0; i < Y.size(); ++i) {
        mean_abs_error += std::abs(Y_prediction[i] - Y[i]);
    }
    mean_abs_error /= Y.size();

    double accuracy = 100.0 - mean_abs_error * 100.0;
    return accuracy;
}

void plot_learning_curve(const vector<float>& costs) {
    std::ofstream dataFile("data.txt");
    if (!dataFile) {
        std::cerr << "Error creating data file." << std::endl;
        return;
    }

    for (int i = 0; i < costs.size(); ++i) {
        dataFile << i * 100 << " " << costs[i] << "\n";
    }
    dataFile.close();

    std::ofstream gnuplotScript("plot.gp");
    if (!gnuplotScript) {
        std::cerr << "Error creating Gnuplot script." << std::endl;
        return;
    }
    
    gnuplotScript << "set title 'Plot of learning curve'\n";
    gnuplotScript << "set xlabel 'number of iteration'\n";
    gnuplotScript << "set ylabel 'cost'\n";
    gnuplotScript << "set grid\n";
    gnuplotScript << "plot 'data.txt' with linespoints title 'learning curve'\n";
    gnuplotScript.close();

    system("gnuplot -p plot.gp");
}

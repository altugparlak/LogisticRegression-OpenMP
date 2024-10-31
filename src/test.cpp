#include "test.h"
#include "utils.h"

void propagation_test1() {
    cout << "____________ Propagation Test - 1 ____________" << endl;
    vector<float> w = {1.0, 2.0};
    vector<float> b = {1.5};

    vector<cv::Mat> X;
    X.push_back((cv::Mat_<float>(1, 3) << 1.0, -2.0, -1.0));
    X.push_back((cv::Mat_<float>(1, 3) << 3.0, 0.5, -3.2));
    vector<int> Y = {1, 1, 0};

    auto result = propagation(w, b, X, Y);

    auto grads = result.first;
    float cost = result.second;
    
    cout << "dw = ";
    for (int i = 0; i < grads["dw"].size(); i++)
    {
        cout << grads["dw"][i] << " ";
    }
    cout << "db = " << grads["db"][0] << endl;
    cout << "cost = " << cost << endl;
    cout << "______________________________________________" << endl;
}

void propagation_test2(const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set) {
    cout << "____________ Propagation Test - 2 ____________" << endl;
    auto result = propagation(w, b, train_set, true_label_set);
     // Access results
    auto grads = result.first;
    float cost = result.second;
    
    // Print the results
    cout << "dw = ";
    for (int i = 0; i < 10; i++)
    {
        cout << grads["dw"][i] << " ";
    }
    cout << " ..." << endl;
    cout << "db = " << grads["db"][0] << endl;
    cout << "cost = " << cost << endl;
    cout << "______________________________________________" << endl;
}

void optimization_test1() {
    cout << "___________ Optimization Test - 1 ____________" << endl;
    vector<float> w = {1.0, 2.0};
    vector<float> b = {1.5};

    vector<cv::Mat> X;
    X.push_back((cv::Mat_<float>(1, 3) << 1.0, -2.0, -1.0));
    X.push_back((cv::Mat_<float>(1, 3) << 3.0, 0.5, -3.2));
    vector<int> Y = {1, 1, 0};
    auto result = optimize(w, b, X, Y, 101, 0.009);
    cout << "______________________________________________" << endl;
}

void optimization_test2(const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate) {
    cout << "___________ Optimization Test - 2 ____________" << endl;
    auto result = optimize(w, b, train_set, true_label_set, num_iterations, learning_rate);
    cout << "______________________________________________" << endl;
}

void prediction_test1() {
    cout << "_____________ Prediction Test - 1 ____________" << endl;
    vector<float> w = {0.11245, 0.23106};
    vector<float> b = {-0.3};

    vector<cv::Mat> X;
    X.push_back((cv::Mat_<float>(1, 3) << 1.0, -1.1, -3.2));
    X.push_back((cv::Mat_<float>(1, 3) << 1.2, 2.0, 0.1));
    auto result = predict(w, b, X);
    cout << "Predictions: " << endl;
    for (int i = 0; i < result.size(); i++)
    {
        cout << result[i] << ", ";
    }
    cout << endl;
    cout << "______________________________________________" << endl;
}

void prediction_test2(const vector<float>& w, const vector<float>& b, 
    const vector<cv::Mat>& train_set, const vector<int>& true_label_set,
    int num_iterations, float learning_rate) {
    cout << "_____________ Prediction Test - 2 ____________" << endl;
    auto result = optimize(w, b, train_set, true_label_set, num_iterations, learning_rate);
    
    auto params = std::get<0>(result);
    auto grads = std::get<1>(result);
    auto costs = std::get<2>(result);

    vector<float> w_0 = params["w"];
    vector<float> b_0 = params["b"];

    // To-Do: Handle prediction
    //Y_prediction_test = predict(w, b, X_test)
    
    cout << "______________________________________________" << endl;
}
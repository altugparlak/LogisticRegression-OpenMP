#include "utils.h"

using namespace std;

const string TRAIN_IMAGE_PATH = "../dataset/Train/";
const string TEST_IMAGE_PATH = "../dataset/Test/";

int main(int argc, char** argv) {
    omp_set_num_threads(100);
    
    // Load images
    vector<cv::Mat> train_images = getImages(TRAIN_IMAGE_PATH);
    cout << "Loaded " << train_images.size() << " train images." << endl;
    cout << "Image shape: " << train_images[0].size() << endl;

    vector<cv::Mat> test_images = getImages(TEST_IMAGE_PATH);
    cout << "Loaded " << test_images.size() << " test images." << endl;
    cout << "Image shape: " << test_images[0].size() << endl;

    // Reshape the images
    vector<cv::Mat> train_flattened_images = getFlattenImages(train_images);
    cout << "Loaded " << train_flattened_images.size() << " flattened train images." << endl;
    cout << "Flatten image shape: " << train_flattened_images[0].size() << endl;
        
    vector<cv::Mat> test_flattened_images = getFlattenImages(test_images);
    cout << "Loaded " << test_flattened_images.size() << " flattened test images." << endl;
    cout << "Flatten image shape: " << test_flattened_images[0].size() << endl;

    // Normalize flattened images
    normalize(train_flattened_images);
    cout << "Normalized " << train_flattened_images.size() << " flattened train images." << endl;
        
    normalize(test_flattened_images);
    cout << "Normalized " << test_flattened_images.size() << " flattened test images." << endl;
    
    cv::Size size = train_images[0].size();
    int label_size = size.width;
    int w_size = size.width * size.height * 3;
    
    cout << "l:" << label_size << " w: " << w_size << endl;
    // Initialize w, b and label values
    //vector<int> true_label_set(label_size, 1);
    //vector<float> w(w_size, 0.0);
    //vector<float> b(w_size, 0.0);
    
    // Initialize weights and bias
    vector<float> w = {1.0, 2.0};  // This should be a vector with two elements
    vector<float> b = {1.5, 1.5};  // Bias needs to be a vector as well

    // Initialize training set (assuming you want to use cv::Mat)
    vector<cv::Mat> X;
    X.push_back((cv::Mat_<float>(1, 3) << 1.0, -2.0, -1.0)); // First example
    X.push_back((cv::Mat_<float>(1, 3) << 3.0, 0.5, -3.2));  // Second example

    // Initialize true labels
    vector<int> Y = {1, 1, 0};  // Adjust the size if needed to match training set
    
    
    auto result = optimize(w, b, X, Y, 100, 0.009);

    /*
    // Access results
    auto grads = result.first;
    float cost = result.second;

    // Print the results
    cout << "dw = ";
    for (const auto& val : grads["dw"]) {
        cout << val << " ";
    }
    cout << endl;

    cout << "db = ";
    for (const auto& val : grads["db"]) {
        cout << val << " ";
    }
    cout << endl;

    cout << "cost = " << cost << endl;
    */
    return 0;
}

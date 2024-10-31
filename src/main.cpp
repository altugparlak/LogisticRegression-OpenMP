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

    //vector<cv::Mat> test_images = getImages(TEST_IMAGE_PATH);
    //cout << "Loaded " << test_images.size() << " test images." << endl;
    //cout << "Image shape: " << test_images[0].size() << endl;
    cout << "================" << endl;
    // Reshape the images
    vector<cv::Mat> train_flattened_images = getFlattenImages(train_images);
    cout << "Loaded " << train_flattened_images.size() << " flattened train images." << endl;
    cout << "Flatten image shape: " << train_flattened_images[0].size() << endl;
    
    // Initialize transposed matrix
    vector<cv::Mat> transposed(12288);
    #pragma omp parallel for
    for (int i = 0; i < transposed.size(); i++) {
        transposed[i] = cv::Mat::zeros(1, train_flattened_images.size(), CV_32F);
    }
    // Fill the transposed matrix
    for (int i = 0; i < train_flattened_images.size(); i++) {
        for (int j = 0; j < 12288; j++) {
            transposed[j].at<int>(i) = train_flattened_images[i].at<int>(j);
        }
    }

    // Loop through each element in the transposed vector
    cout << "transposed size x: " << transposed.size() << endl;
    cout << "transposed size y: " << transposed[0].size() << endl;
 
    //vector<cv::Mat> test_flattened_images = getFlattenImages(test_images);
    //cout << "Loaded " << test_flattened_images.size() << " flattened test images." << endl;
    //cout << "Flatten image shape: " << test_flattened_images[0].size() << endl;

    // Normalize flattened images
    //normalize(train_flattened_images);
    //cout << "Normalized " << train_flattened_images.size() << " flattened train images." << endl;
        
    //normalize(test_flattened_images);
    //cout << "Normalized " << test_flattened_images.size() << " flattened test images." << endl;
    
    cv::Size size = train_images[0].size();
    int label_size = size.width;
    int w_size = size.width * size.height * 3;
    
    cout << "l:" << label_size << " w: " << w_size << endl;
    // Initialize w, b and label values
    vector<int> true_label_set(train_flattened_images.size(), 1);
    
    vector<float> w(w_size, 0.0);
    vector<float> b(1, 0.0);
    
    // Initialize weights and bias
    vector<float> w2 = {1.0, 2.0};  // This should be a vector with two elements
    vector<float> b2 = {1.5};  // Bias needs to be a vector as well

    // Initialize training set (assuming you want to use cv::Mat)
    vector<cv::Mat> X;
    X.push_back((cv::Mat_<float>(1, 3) << 1.0, -2.0, -1.0)); // First example
    X.push_back((cv::Mat_<float>(1, 3) << 3.0, 0.5, -3.2));  // Second example

    // Initialize true labels
    vector<int> Y = {1, 1, 0};  // Adjust the size if needed to match training set
    

    //auto result = optimize(w, b, train_flattened_images, true_label_set, 100, 0.009);
    //auto result = optimize(w2, b2, X, Y, 100, 0.009);
    auto result = propagation(w, b, transposed, true_label_set);
    //auto result = propagation(w2, b2, X, Y);
    
    // Access results
    auto grads = result.first;
    float cost = result.second;
    
    // Print the results
    cout << "dw = ";
    for (int i = 0; i < 10; i++)
    {
        cout << grads["dw"][i] << " ";
    }
    cout << endl;
    cout << "db = " << grads["db"][0] << endl;
    cout << "cost = " << cost << endl;
    
    //auto result2 = optimize(w, b, train_flattened_images, true_label_set, 100, 0.009);
    //auto result2 = optimize(w2, b2, X, Y, 100, 0.009);
    return 0;
}

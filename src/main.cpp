#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <vector>

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

    return 0;
}

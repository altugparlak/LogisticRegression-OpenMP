#include <stdio.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <vector>

using namespace std;

const string IMAGE_PATH = "../dataset/Train/";

int main(int argc, char** argv) {
    omp_set_num_threads(1);
    
    #pragma omp parallel
    {
        printf("Hello from process: %d\n", omp_get_thread_num());
    }

    #pragma omp single
    {   
        /*
        double value = 0.5;
        double sig = sigmoid(value);
        printf("Sigmoid of %lf is %lf\n", value, sig);
        */
        
        vector<cv::Mat> images = getImages(IMAGE_PATH);
        cout << "Loaded " << images.size() << " images." << endl;
        cout << "Image shape: " << images[0].size() << endl;

        vector<cv::Mat> flatten_images = getFlattenImages(images);
        cout << "Loaded " << flatten_images.size() << " images." << endl;
        cout << "Flatten image shape: " << flatten_images[0].size() << endl;
        
    }

    return 0;
}

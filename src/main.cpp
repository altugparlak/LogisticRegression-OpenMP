#include <stdio.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

int main(int argc, char** argv) {
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        printf("Hello from process: %d\n", omp_get_thread_num());
    }

    #pragma omp single
    {
        double value = 0.5;
        double sig = sigmoid(value);
        printf("Sigmoid of %lf is %lf\n", value, sig);

        std::string image_path = "../dataset/Train/dog.1.jpg";

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if(image.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
        }

        cv::imshow("Loaded Image", image);
        cv::waitKey(0);
    }

    return 0;
}

#include "utils.h"
#include "test.h"

using namespace std;

const string TRAIN_IMAGE_PATH = "../dataset/Train/";
const string TEST_IMAGE_PATH = "../dataset/Test/";

int main(int argc, char** argv) {
    omp_set_num_threads(100);
    
    /* ------------------------------- Handle Data ------------------------------ */
    // Load images
    vector<cv::Mat> train_images = getImages(TRAIN_IMAGE_PATH);
    cout << "Loaded " << train_images.size() << " train images." << endl;
    cout << "Image shape: " << train_images[0].size() << endl;

    cout << "================" << endl;

    cv::Size size = train_images[0].size();
    int w_size = size.width * size.height * 3;
    cout << "Image width:" << size.width << " Flattened image width: " << w_size << endl;
    //vector<cv::Mat> test_images = getImages(TEST_IMAGE_PATH);
    //cout << "Loaded " << test_images.size() << " test images." << endl;
    //cout << "Image shape: " << test_images[0].size() << endl;
    cout << "================" << endl;
    // Reshape the images
    vector<cv::Mat> train_flattened_images = getFlattenImages(train_images);
    cout << "Loaded " << train_flattened_images.size() << " flattened train images." << endl;
    cout << "Flatten image shape: " << train_flattened_images[0].size() << endl;
    
    // Initialize transposed matrix
    vector<cv::Mat> transposed(w_size);
    #pragma omp parallel for
    for (int i = 0; i < transposed.size(); i++) {
        transposed[i] = cv::Mat::zeros(1, train_flattened_images.size(), CV_32F);
    }
    // Fill the transposed matrix
    for (int i = 0; i < train_flattened_images.size(); i++) {
        for (int j = 0; j < transposed.size(); j++) {
            transposed[j].at<int>(i) = train_flattened_images[i].at<int>(j);
        }
    }
 
    //vector<cv::Mat> test_flattened_images = getFlattenImages(test_images);
    //cout << "Loaded " << test_flattened_images.size() << " flattened test images." << endl;
    //cout << "Flatten image shape: " << test_flattened_images[0].size() << endl;

    // Initialize true label set size of input images
    vector<int> true_label_set(train_images.size(), 1);
    cout << "true label size: " << true_label_set.size() << endl;
    // None-dog images: 0
    for (int i = 210; i < true_label_set.size(); i++)
        true_label_set[i] = 0;

    vector<float> w(w_size, -0.2f);
    vector<float> b = {0.0};
    /* -------------------------------------------------------------------------- */
    /* ------------------------------ Test Section ------------------------------ */

    //propagation_test1();
    //propagation_test2(w, b, transposed, true_label_set);
    //optimization_test1();
    optimization_test2(w, b, transposed, true_label_set, 301, 0.009);
    //prediction_test1();

    /* -------------------------------------------------------------------------- */
    
    return 0;
}

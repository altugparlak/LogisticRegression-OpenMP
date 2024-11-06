#include "utils.h"
#include "test.h"

using namespace std;

const string TRAIN_IMAGE_PATH = "../dataset/Train/";
const string TEST_IMAGE_PATH = "../dataset/Test/";

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <num_threads> <num_iterations> <learning_rate>" << endl;
        return EXIT_FAILURE;
    }

    int num_threads = stoi(argv[1]);
    int num_iterations = stoi(argv[2]);
    float learning_rate = stof(argv[3]);

    omp_set_num_threads(num_threads);
    cout << "Threads: " << num_threads << endl 
    << "Iterations: " << num_iterations << endl
    << "Learning Rate: " << learning_rate << endl;
    cout << "================" << endl;
    auto start_time = high_resolution_clock::now();
    /* ----------------------------- Get Input Data ----------------------------- */
    // Load images
    vector<cv::Mat> train_images = getImages(TRAIN_IMAGE_PATH);
    cout << "Loaded " << train_images.size() << " train images." << endl;
    cout << "Image shape: " << train_images[0].size() << endl;
    
    cv::Size size = train_images[0].size();
    int w_size = size.width * size.height * 3;
    cout << "Image width:" << size.width << " Flattened image width: " << w_size << endl;
    cout << "================" << endl;
    vector<cv::Mat> test_images = getImages(TEST_IMAGE_PATH);
    cout << "Loaded " << test_images.size() << " test images." << endl;
    cout << "Image shape: " << test_images[0].size() << endl;
    cout << "================" << endl;
    // Reshape the images
    vector<cv::Mat> train_flattened_images = getFlattenImages(train_images);
    cout << "Loaded " << train_flattened_images.size() << " flattened train images." << endl;
    cout << "Flatten image shape: " << train_flattened_images[0].size() << endl;
    vector<cv::Mat> test_flattened_images = getFlattenImages(test_images);
    cout << "Loaded " << test_flattened_images.size() << " flattened test images." << endl;
    cout << "Flatten image shape: " << test_flattened_images[0].size() << endl;
    cout << "================" << endl;
    /* -------------------------------------------------------------------------- */
    /* ---------------------------- Matrix transpose ---------------------------- */
    // Initialize transposed matrix
    vector<cv::Mat> transposed_train_set(w_size);
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < transposed_train_set.size(); i++) {
        transposed_train_set[i] = cv::Mat::zeros(1, train_flattened_images.size(), CV_32F);
    }
    // Fill the transposed matrix
    for (int i = 0; i < train_flattened_images.size(); i++) {
        for (int j = 0; j < transposed_train_set.size(); j++) {
            transposed_train_set[j].at<int>(i) = train_flattened_images[i].at<int>(j);
        }
    }

    // Initialize transposed matrix
    vector<cv::Mat> transposed_test_set(w_size);
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < transposed_test_set.size(); i++) {
        transposed_test_set[i] = cv::Mat::zeros(1, test_flattened_images.size(), CV_32F);
    }
    // Fill the transposed matrix
    for (int i = 0; i < test_flattened_images.size(); i++) {
        for (int j = 0; j < transposed_test_set.size(); j++) {
            transposed_test_set[j].at<int>(i) = test_flattened_images[i].at<int>(j);
        }
    }
    /* -------------------------------------------------------------------------- */
    /* ------------------------------- True labels ------------------------------ */
    // Initialize true label set size of input images
    vector<int> train_true_label_set(train_images.size(), 1);
    cout << "true label size: " << train_true_label_set.size() << endl;
    // None-dog images: 0
    #pragma omp for schedule(dynamic)
    for (int i = 210; i < train_true_label_set.size(); i++)
        train_true_label_set[i] = 0;

    // Initialize true label set size of input images
    vector<int> test_true_label_set(test_images.size(), 1);
    cout << "true label size: " << test_true_label_set.size() << endl;
    // None-dog images: 0
    for (int i = 50; i < test_true_label_set.size(); i++)
        test_true_label_set[i] = 0;
    
    /* -------------------------------------------------------------------------- */
    vector<float> w(w_size, 0.0f);
    vector<float> b = {0.0};
    /* -------------------------------------------------------------------------- */
    /* ------------------------------ Test Section ------------------------------ */

    //propagation_test1();
    //propagation_test2(w, b, transposed, true_label_set);
    //optimization_test1();
    //optimization_test2(w, b, transposed_train_set, train_true_label_set, num_iterations, learning_rate);
    //prediction_test1();
    prediction_test2(w, b, transposed_train_set, train_true_label_set, transposed_test_set, test_true_label_set, num_iterations, learning_rate);

    /* -------------------------------------------------------------------------- */
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);

    cout << "Program completed in " << duration.count() << " seconds." << endl;

    return EXIT_SUCCESS;
}

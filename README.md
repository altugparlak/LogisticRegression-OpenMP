# LogisticRegression-OpenMP

<!-- ABOUT THE PROJECT -->
## About The Project
This project implements logistic regression using the OpenMP library, aiming to achieve efficient, multithreaded learning operations. It leverages both `openmp` for parallel processing and `opencv` for image handling and data manipulation.

### Built With
* [![C++][C++-logo]][C++-url]

## Prerequisites
Ensure that OpenCV and OpenMP are installed before building the project:

- **OpenCV**: Follow the [OpenCV installation instructions](https://opencv.org/releases/) for your platform.
- **OpenMP**: This is usually installed by default with most compilers. If not, refer to the [OpenMP installation instructions](https://www.openmp.org/resources/openmp-compilers-tools/).

<!-- HOW TO BUILD -->
## How to Build
1. Clone the repository:
   ```sh
   git clone https://github.com/altugparlak/LogisticRegression-OpenMP.git
   ```
2. Navigate to the project directory:
   ```sh
   cd LogisticRegression-OpenMP
   ```
3. Navigate to the build directory and compile with CMake:
   ```sh
   cd build
   make
   ```
4. Run the application:
   ```sh
   ./main.exe <num_thread> <num_iterations> <learning_rate>
   ```
   
<!-- RESULTS -->
## Results
```
Threads: 10
Iterations: 3001
Learning Rate: 0.005
================
Loaded 261 train images.
Image shape: [64 x 64]
Image width:64 Flattened image width: 12288
================
Loaded 70 test images.
Image shape: [64 x 64]
================
Loaded 261 flattened train images.
Flatten image shape: [12288 x 1]
Loaded 70 flattened test images.
Flatten image shape: [12288 x 1]
================
true label size: 261
true label size: 70
_____________ Prediction Test - 2 ____________
Cost after iteration 0: 18.5266
Cost after iteration 100: 5.89726
Cost after iteration 200: 5.7179
Cost after iteration 300: 5.68837
Cost after iteration 400: 5.65021
Cost after iteration 500: 5.42075
Cost after iteration 600: 5.24648
Cost after iteration 700: 5.04455
Cost after iteration 800: 4.92018
Cost after iteration 900: 4.78962
Cost after iteration 1000: 4.52974
Cost after iteration 1100: 4.23881
Cost after iteration 1200: 3.83913
Cost after iteration 1300: 3.4368
Cost after iteration 1400: 3.05164
Cost after iteration 1500: 2.63407
Cost after iteration 1600: 2.2617
Cost after iteration 1700: 2.01017
Cost after iteration 1800: 1.45863
Cost after iteration 1900: 1.11906
Cost after iteration 2000: 0.785798
Cost after iteration 2100: 0.601083
Cost after iteration 2200: 0.532884
Cost after iteration 2300: 0.391027
Cost after iteration 2400: 0.30409
Cost after iteration 2500: 0.300219
Cost after iteration 2600: 0.282101
Cost after iteration 2700: 0.195868
Cost after iteration 2800: 0.177084
Cost after iteration 2900: 0.185942
Cost after iteration 3000: 0.189521
Train accuracy: 95.4023 %
Test accuracy: 65.7143 %
______________________________________________
Program completed in 95 seconds.
```

<!-- REQUIREMENTS -->
## Requirements
* `OpenCV`
* `OpenMP`

<!-- Badge Images -->
[C++-logo]: https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white

<!-- Badge URLs -->
[C++-url]: https://isocpp.org/

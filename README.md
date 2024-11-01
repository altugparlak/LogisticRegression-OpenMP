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
Threads: 120
Iterations: 2501
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
Cost after iteration 100: 5.83352
Cost after iteration 200: 5.84843
Cost after iteration 300: 5.96152
Cost after iteration 400: 5.66537
Cost after iteration 500: 5.84278
Cost after iteration 600: 5.57628
Cost after iteration 700: 5.42906
Cost after iteration 800: 5.33973
Cost after iteration 900: 5.19019
Cost after iteration 1000: 4.87878
Cost after iteration 1100: 4.73885
Cost after iteration 1200: 4.31415
Cost after iteration 1300: 3.86874
Cost after iteration 1400: 3.27671
Cost after iteration 1500: 2.75204
Cost after iteration 1600: 2.43861
Cost after iteration 1700: 1.79153
Cost after iteration 1800: 1.20521
Cost after iteration 1900: 0.879771
Cost after iteration 2000: 0.674454
Cost after iteration 2100: 0.648342
Cost after iteration 2200: 0.411896
Cost after iteration 2300: 0.322047
Cost after iteration 2400: 0.294056
Cost after iteration 2500: 0.256254
Train accuracy: 91.1877 %
Test accuracy: 61.4286 %
______________________________________________
Program completed in 158 seconds.
```

<!-- REQUIREMENTS -->
## Requirements
* `OpenCV`
* `OpenMP`

<!-- Badge Images -->
[C++-logo]: https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white

<!-- Badge URLs -->
[C++-url]: https://isocpp.org/

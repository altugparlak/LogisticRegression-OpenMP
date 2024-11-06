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
Iterations: 5501
Learning Rate: 0.0021
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
Cost after iteration 0: 0.693147
Cost after iteration 100: 0.453358
Cost after iteration 200: 0.424136
Cost after iteration 300: 0.401257
Cost after iteration 400: 0.385049
Cost after iteration 500: 0.365592
Cost after iteration 600: 0.354807
Cost after iteration 700: 0.344988
Cost after iteration 800: 0.327178
Cost after iteration 900: 0.324203
Cost after iteration 1000: 0.300126
Cost after iteration 1100: 0.294407
Cost after iteration 1200: 0.29711
Cost after iteration 1300: 0.289715
Cost after iteration 1400: 0.265431
Cost after iteration 1500: 0.250932
Cost after iteration 1600: 0.25756
Cost after iteration 1700: 0.24728
Cost after iteration 1800: 0.259737
Cost after iteration 1900: 0.239741
Cost after iteration 2000: 0.222708
Cost after iteration 2100: 0.222294
Cost after iteration 2200: 0.229368
Cost after iteration 2300: 0.222064
Cost after iteration 2400: 0.203185
Cost after iteration 2500: 0.204815
Cost after iteration 2600: 0.200447
Cost after iteration 2700: 0.205646
Cost after iteration 2800: 0.196681
Cost after iteration 2900: 0.191428
Cost after iteration 3000: 0.190911
Cost after iteration 3100: 0.17661
Cost after iteration 3200: 0.167813
Cost after iteration 3300: 0.182509
Cost after iteration 3400: 0.183065
Cost after iteration 3500: 0.173664
Cost after iteration 3600: 0.161989
Cost after iteration 3700: 0.166616
Cost after iteration 3800: 0.151963
Cost after iteration 3900: 0.161444
Cost after iteration 4000: 0.155389
Cost after iteration 4100: 0.154951
Cost after iteration 4200: 0.151035
Cost after iteration 4300: 0.145544
Cost after iteration 4400: 0.150318
Cost after iteration 4500: 0.163328
Cost after iteration 4600: 0.131378
Cost after iteration 4700: 0.139887
Cost after iteration 4800: 0.132558
Cost after iteration 4900: 0.135125
Cost after iteration 5000: 0.143895
Cost after iteration 5100: 0.145502
Cost after iteration 5200: 0.120616
Cost after iteration 5300: 0.125176
Cost after iteration 5400: 0.127477
Cost after iteration 5500: 0.117866
Train accuracy: 97.7011 %
Test accuracy: 68.5714 %
______________________________________________
Program completed in 158 seconds.
```
![iter5500-lr0,0021](https://github.com/user-attachments/assets/693b7352-4e79-4ae9-a1ec-e769cb1d9ff0)

<!-- Badge Images -->
[C++-logo]: https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white

<!-- Badge URLs -->
[C++-url]: https://isocpp.org/

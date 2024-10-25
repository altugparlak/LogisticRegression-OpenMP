#include <stdio.h>
#include <omp.h>
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
    }

    return 0;
}

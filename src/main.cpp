#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv){
    omp_set_num_teams(10);
    
    #pragma omp parallel
    {
        printf("Hello from process: %d\n", omp_get_thread_num());
    }

    return 0;
}
